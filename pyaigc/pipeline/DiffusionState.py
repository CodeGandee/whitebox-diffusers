# stable diffusion with control net
import torch
import numpy as np
import numpy.typing as npt
from typing import ClassVar, Literal, Self
from attrs import define, field
import attrs.validators as av
import attrs.setters as aset

from pyaigc.model.ControlnetBase import IControlnet
from pyaigc.model.DiffusionScheduler import DiffusionScheduler
import pyaigc.GlobalConfig as C

@define(kw_only=True)
class ControlNetInfo:
    model : IControlnet
    name : str | None = None
    weight : float = 1.0
    
@define(kw_only=True)
class TextPromptData:
    text : str = ""
    embedding : torch.Tensor | None = None  # (n_token, n_feature_dim) word embedding
    projected_embedding : torch.Tensor | None = None  # (1, n_feature_dim) 1d embedding
    weight : float = 1.0
    
@define(kw_only=True)
class ImagePromptData:
    ''' for use with ip adapter
    '''
    name : str = field()    # name of the ip adapter
    image : C.ImageTensor = field(default=None)
    positive_embedding : torch.Tensor = field(default=None)
    negative_embedding : torch.Tensor = field(default=None)

@define(kw_only=True)
class ControlnetData:
    # 4d tensor, in the form of (batch_size, channel, height, width)
    image_data : torch.Tensor = field(converter=lambda x : x if isinstance(x, torch.Tensor) else torch.from_numpy(x))
    image_data_layout : Literal['nhwc', 'nchw'] # layout of image_data
    weight : float = 1.0    # weight of this control signal
    
    @property
    def width(self)->int:
        if self.image_data_layout == 'nhwc':
            return self.image_data.shape[2]
        elif self.image_data_layout == 'nchw':
            return self.image_data.shape[3]
    
    @property
    def height(self)->int:
        if self.image_data_layout == 'nhwc':
            return self.image_data.shape[1]
        elif self.image_data_layout == 'nchw':
            return self.image_data.shape[2]
        
    @image_data.validator
    def _image_data_validator(self, attribute, value : torch.Tensor):
        assert len(value.shape) == 4, 'image_data must be a 4d tensor'
    
    # def __attrs_post_init__(self):
    #     if len(self.image_data.shape) != 4:
    #         self.image_data : torch.Tensor = to_4d_tensor(self.image_data, 
    #                                                     input_layout=self.image_data_layout, 
    #                                                     output_layout=self.image_data_layout)
    
def cvt_generator_to_state(generator : torch.Generator | torch.Tensor) -> torch.Tensor:
    ''' convert a torch.Generator to its state, which is a torch.Tensor
    '''
    if generator is None:
        return None
    if isinstance(generator, torch.Generator):
        return generator.get_state()
    elif isinstance(generator, torch.Tensor):
        return generator
    else:
        raise TypeError(f'generator must be a torch.Generator or torch.Tensor, got {type(generator)}')
    
@define(kw_only=True)
class DiffusionState:
    ''' input, output and intermediate states of diffusion model
    '''
    DefaultNumberOfInferenceTimesteps : ClassVar[int] = 20
    
    target_width : int = field()  # target image size, must be divisible by 8
    target_height : int = field()  # target image size, must be divisible by 8
        
    # prompts
    positive_prompt : TextPromptData = field(factory=TextPromptData)
    negative_prompt : TextPromptData = field(factory=TextPromptData)
    
    # the current latent
    latent : torch.Tensor | None = field(default=None)
    
    # for image-to-image, latent for the clean image before adding noise
    latent_clean : torch.Tensor | None = field(default=None)
    
    # only apply denoise within this mask
    # same size as latent, float value between 0 and 1, 1 means fully denoise, 0 means no change
    latent_denoise_mask : torch.Tensor | None = field(default=None)
    
    # classifier-free guidence scale, the positive value leans the image towards the positive prompt, and vice versa
    guidance_scale : float = 7.5    # following the original paper
    
    # the current step index, in the sequence of timestep_indices
    # this step is already denoised. If None, then the diffusion has not started yet.
    # this information is also stored in scheduler.current_step_index, but we keep it here to allow one to set the step index without the scheduler
    # during denoise, you need to take care of the consistency between this and scheduler.current_step_index
    current_step_index : int | None = None
    
    # denoise timesteps in terms of their indices in the full timestep sequence
    # let k=timestep_index_sequence[i], it means that at step i, the denoise timestep is scheduler.all_timesteps[k]
    # timestep_index_sequence : np.ndarray | None = field(default=None, converter=lambda x : np.asarray(x, dtype=int) if x is not None else None)
    timestep_index_sequence : np.ndarray | None = field(default=None)
    
    # controlnet data
    controlnet_data : dict[str, ControlnetData] = field(factory=dict)
    
    # ip adapter data, indexed by the names of the ip adapters
    image_prompt_data : dict[str, ImagePromptData] = field(factory=dict)
    
    # random generator
    rng : torch.Generator | None = field(default=None, validator=av.optional(av.instance_of(torch.Generator)))
    
    # denoise scheduler
    scheduler : DiffusionScheduler | None = None
    
    def clone(self) -> Self:
        import copy
        
        # make a shallow copy and remove uncloneable fields
        obj = copy.copy(self)
        obj.rng = None
        obj = copy.deepcopy(obj)
        
        if self.rng is not None:
            obj.rng = torch.Generator(device=self.rng.device).set_state(self.rng.get_state())
        
        return obj
    
    @property
    def is_inpainting(self) -> bool:
        ''' we are denoising in inpainting mode?
        '''
        return self.latent_clean is not None
    
    @property
    def is_masked_denoising(self) -> bool:
        ''' we are only denoising part of the latent?
        '''
        return self.latent_denoise_mask is not None
    
    def __attrs_post_init__(self): 
        if self.timestep_index_sequence is None and self.scheduler is not None:
            max_step = len(self.scheduler.all_selectable_timesteps)
            n_step = self.DefaultNumberOfInferenceTimesteps
            idx_timesteps = np.linspace(0, max_step, n_step, endpoint=False, dtype=int)[::-1]
            assert np.unique(idx_timesteps).size == idx_timesteps.size, 'timestep_sequence must be unique'
            self.set_timestep_indices(idx_timesteps)
            
        if self.latent is not None and self.latent_clean is not None:
            # must have the same shape
            assert self.latent.shape == self.latent_clean.shape, 'latent and latent_clean must have the same shape'
            
        if self.latent_denoise_mask is not None:
            for lt in (self.latent, self.latent_clean):
                # latent and latent must must have the same shape
                if lt is not None:
                    assert self.latent_denoise_mask.shape == lt.shape, 'latent and latent_denoise_mask must have the same shape'
            
    @property
    def num_all_timesteps(self) -> int | None:
        if self.timestep_index_sequence is None:
            return 0
        else:
            return len(self.timestep_index_sequence)
        
    @property
    def num_remaining_denosie_step(self) -> int:
        ''' number of remaining denoise steps, 0 means the diffusion denoise has finished
        '''
        if self.timestep_index_sequence is None:
            return 0
        
        if self.current_step_index is None:
            return self.num_all_timesteps
        
        return self.num_all_timesteps - self.current_step_index - 1
        
    @property
    def max_timestep_index(self) -> int | None:
        if self.scheduler is None:
            return None
        return len(self.scheduler.all_selectable_timesteps) - 1
        
    @property
    def random_generator(self) -> torch.Generator:
        ''' get the random generator based on the random_generator_state, or the default generator if the state is None
        '''
        # if self.rng_state is None:
        #     return torch.default_generator
        # else:
        #     return torch.Generator(device=self.rng_device).set_state(self.rng_state)
        
        if self.rng is None:
            return torch.default_generator
        else:
            return self.rng
        
    # validators and defaults
    @target_width.validator
    def _target_width_validator(self, attribute, value : int):
        assert value % 8 == 0, 'target_width must be divisible by 8'
        
    @target_height.validator
    def _target_height_validator(self, attribute, value : int):
        assert value % 8 == 0, 'target_height must be divisible by 8'
        
    @timestep_index_sequence.validator
    def _timestep_index_sequence_validator(self, attribute, value : npt.NDArray[np.int_] | None):
        if value is None:
            return
        assert len(value) > 0, 'timestep_index_sequence must be non-empty'
        assert np.all(np.diff(value) <= 0), 'timestep_index_sequence must be in descending order'

    # TODO: remove this        
    # def set_random_generator_state(self, rng_state : torch.Tensor, rng_device : str | torch.device):
    #     ''' set the random generator state, which can be either a torch.Generator or its state (a torch.Tensor)
    #     '''
    #     self.rng_state = rng_state.clone()
    #     self.rng_device = str(rng_device)
        
    def set_current_step_index(self, idx : int | None):
        ''' set the current step index, which must be less than num_timesteps, None means the diffusion has not started yet
        '''
        
        if idx is not None:
            assert idx < self.num_all_timesteps, 'idx must be less than num_timesteps'
            
        self.current_step_index = idx
        
        # adjust scheduler
        if self.scheduler is not None:
            self.scheduler.set_current_step_index(idx)
            
    def set_timestep_indices(self, indices : np.ndarray):
        ''' set timestep_sequence by specifying the indices of the timesteps
        '''
        assert len(indices) > 0, 'indices must be non-empty'
        assert indices.max() <= self.max_timestep_index, 'indices must be no more than max_timestep_index'
        
        self.timestep_index_sequence = indices
        
        if self.scheduler is not None:
            self.scheduler.set_selected_timesteps_by_index(indices)
            self.scheduler.set_current_step_index(self.current_step_index)
    
    def set_timestep_sequence_by_num_steps(self, num_steps : int, max_timestep_index : int = None):
        ''' set timestep_sequence by specifying the number of steps, which uses the default max_timestep
        as the timestep upper bound.
        
        parameters
        ------------
        num_steps : int
            number of steps
        max_timestep_index : int | None
            the maximum timestep index, if None, then use the default max_timestep_index in scheduler
        '''
        assert num_steps > 0, 'num_steps must be positive'
        assert self.scheduler is not None, 'scheduler must be specified'
        tmax = max_timestep_index
        if tmax is None:
            tmax = self.max_timestep_index
        
        idx = np.linspace(0, tmax, num_steps, endpoint=True, dtype=int)[::-1]
        self.set_timestep_indices(idx)
        
    def get_latent_for_inpainting(self, next_timestep_index : int = None,
                                  latent_noise : torch.Tensor | None = None) -> torch.Tensor:
        ''' get the latent tensor for inpainting use, which is blended with a (noise + clean latent) and the current latent
        
        parameters
        -------------
        next_timestep_index : int | None
            the timestep index for the next denoise step, noise will be added to the known region of the latent, to the next_timestep_index, 
            and then blended with the current latent. 
            If -1, then we assume the current latent is fully denoised.
            If None, then we get this information from scheduler
        '''
        dstate = self
        
        assert dstate.latent is not None, 'latent must exist'
        assert dstate.latent_clean is not None, 'latent for clean image must exist'
        assert dstate.scheduler is not None, 'scheduler must exist'    
        
        # if denoise_mask is not given, we assume that latent should be denoised everywhere
        if dstate.latent_denoise_mask is None:
            return dstate.latent
        
        torch_dtype = self.latent.dtype
        torch_device = self.latent.device
        
        # now we have clean latent and denoise mask, we need to blend them with current latent
        
        # add noise to clean latent, up to the next timestep index
        if latent_noise is None:
            noise = torch.randn(dstate.latent.shape, generator=dstate.random_generator)
        else:
            noise = latent_noise.to(dtype=torch_dtype, device=torch_device)
        mask_known = 1 - dstate.latent_denoise_mask if dstate.latent_denoise_mask is not None else None
        mask_unknown = dstate.latent_denoise_mask if dstate.latent_denoise_mask is not None else None
        
        # are we at the last denosing step? if so, we should blend with the clean latent
        
        if next_timestep_index is None:
            next_step_index = dstate.scheduler.next_step_index
            
            if next_step_index is None:
                next_timestep_index = -1 # fully denoised
            else:
                next_timestep_index = dstate.timestep_index_sequence[next_step_index]
                
        # the noise for dstate.scheduler.current_step_index is already denoised, now we are with noise for next_step_index
        if next_timestep_index < 0: # nearly fully denoised, blend with clean latent
            latent_known_region = dstate.latent_clean
        else:
            # latent for known region, where latent_mask[...,i,j] = 0
            latent_known_region = dstate.add_masked_noise_to_given_latent(latent = dstate.latent_clean, 
                                                                          noise = noise, 
                                                                          to_timestep_index=next_timestep_index,
                                                                          latent_mask = mask_known)
            
        latent_known_region = latent_known_region.to(device=torch_device, dtype=torch_dtype)
        mask_unknown = mask_unknown.to(device=torch_device, dtype=torch_dtype)
        latent_blended = torch.lerp(latent_known_region, dstate.latent, mask_unknown)
        return latent_blended
        
    def add_masked_noise_to_given_latent(self, latent: torch.Tensor, noise : torch.Tensor, 
                                         to_timestep_index : int, 
                                         latent_mask : torch.Tensor | None = None,
                                         apply_latent_mask : bool = True) -> torch.Tensor:
        ''' add noise to a given latent, same shape as self.latent, with mask applied
        
        parameters
        -------------
        latent : torch.Tensor
            the latent to add noise to, must have the same shape as self.latent
        noise : torch.Tensor
            the noise to be added, must have the same shape as latent, assumed to be sampled from unit Gaussian
        to_timestep_index : int
            after adding the noise, the latent will be at this timestep index. Must be in range [0, max_timestep_index]
        latent_mask : torch.Tensor | None
            the mask that defines the region to add noise, float mask within (0,1), the larger the value, the more noise will be added.
            If None, then use latent_denoise_mask
        apply_latent_mask : bool
            whether the noise will be applied to the region defined by the latent_mask.
            If False, no matter what the latent_mask is, the noise will be added to the whole latent.
            
        return
        ----------
        latent_with_noise : torch.Tensor
            the latent with noise added, with mask applied
        '''
        # scheduler is required
        assert self.scheduler is not None, 'scheduler must be specified'
        
        # latent must have the same shape as self.latent
        assert latent is not None, 'latent must be given'
        assert latent.shape == self.latent.shape, 'latent must have the same shape as self.latent'
        
        # timestep index must be in range
        assert to_timestep_index >= 0 and to_timestep_index <= self.max_timestep_index, 'to_timestep_index must be in range'
        
        # use latent_denoise_mask if latent_mask is None
        if latent_mask is None:
            latent_mask = self.latent_denoise_mask
        else:
            assert latent_mask.shape == self.latent.shape, 'noise_mask must have the same shape as latent'
            
        noise = noise.to(dtype=self.latent.dtype, device=self.latent.device)
        latent_with_noise = self.scheduler.add_noise(latent, noise, to_timestep_index = to_timestep_index)
        
        # blend it with the original latent
        output : torch.Tensor = None
        if latent_mask is None or not apply_latent_mask:
            output = latent_with_noise
        else:
            latent_mask = latent_mask.to(dtype=self.latent.dtype, device=self.latent.device)
            latent_blended : torch.Tensor = torch.lerp(latent, latent_with_noise, latent_mask)
            output = latent_blended
        output = output.to(dtype=latent.dtype, device=latent.device)
        return output
        
    def add_masked_noise_to_latent(self, noise : torch.Tensor, 
                                   to_timestep_index : int, 
                                   latent_mask : torch.Tensor | None = None,
                                   apply_latent_mask : bool = True):
        ''' add noise to self.latent, with mask applied
        
        parameters
        -------------
        noise : torch.Tensor
            the noise to be added, must have the same shape as latent, assumed to be sampled from unit Gaussian
        to_timestep_index : int
            after adding the noise, the latent will be at this timestep index. Must be in range [0, max_timestep_index]
        latent_mask : torch.Tensor | None
            the mask that defines the region to add noise, float mask within (0,1), the larger the value, the more noise will be added.
            If None, then use latent_denoise_mask
        apply_latent_mask : bool
            whether the noise will be applied to the region defined by the latent_mask.
            If False, no matter what the latent_mask is, the noise will be added to the whole latent.
        '''
        
        # scheduler is required
        assert self.scheduler is not None, 'scheduler must be specified'
        
        # latent must exist
        assert self.latent is not None, 'latent must exist'
        
        # timestep index must be in range
        assert to_timestep_index >= 0 and to_timestep_index <= self.max_timestep_index, 'to_timestep_index must be in range'
        
        # use latent_denoise_mask if latent_mask is None
        if latent_mask is None:
            latent_mask = self.latent_denoise_mask
        else:
            assert latent_mask.shape == self.latent.shape, 'noise_mask must have the same shape as latent'
            
        latent_new = self.add_masked_noise_to_given_latent(self.latent, noise, to_timestep_index, latent_mask, apply_latent_mask)
        self.latent = latent_new
            
        # noise = noise.to(dtype=self.latent.dtype, device=self.latent.device)
        # latent_with_noise = self.scheduler.add_noise(self.latent, noise, to_timestep_index = to_timestep_index)
        
        # # blend it with the original latent
        # if latent_mask is None or not apply_latent_mask:
        #     self.latent = latent_with_noise
        # else:
        #     latent_mask = latent_mask.to(dtype=self.latent.dtype, device=self.latent.device)
        #     self.latent = torch.lerp(self.latent, latent_with_noise, latent_mask)
        
            