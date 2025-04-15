# stable diffusion with control net
import torch
import numpy as np

from pyaigc.model.DiffusionTextModelBase import IDiffusionTextModel
from pyaigc.model.DiffusionUnetBase import IDiffusionUnet
from pyaigc.model.DiffusionVaeBase import IDiffusionVae
from pyaigc.model.ControlnetBase import IControlnet, ControlnetModelOutput
from pyaigc.model.DiffusionScheduler import DiffusionScheduler, SupportedSchedulerType
from pyaigc.helper.ip_adapter import (
    IPAdapterHelper
)

from pyaigc.pipeline.DiffusionState import (
    DiffusionState,
    ControlnetData,
    ControlNetInfo,
    DiffusionScheduler,
    TextPromptData,
    ImagePromptData
)
from pyaigc.model.ImageEncoderBase import IImageEncoder

import pyaigc.GlobalConfig as C

from igpy.common.shortfunc import to_4d_tensor
class StableDiffusionWithControl:
    
    # if the weight is lower than this, the controlnet is skipped
    ControlnetMinimumWeight : float = 1e-4
    
    def __init__(self) -> None:
        super().__init__()
        self.m_text_model : IDiffusionTextModel = None
        self.m_unet : IDiffusionUnet = None
        self.m_vae : IDiffusionVae = None
        
        # if upcast is used, then this is the upcasted version of vae
        self.m_vae_fp32 : IDiffusionVae = None
        
        # control nets indexed by user defined name
        self.m_controlnets : dict[str, ControlNetInfo] = {}
        
        # scheduler
        self.m_scheduler : DiffusionScheduler = None
        
        # image encoder for ip adapter
        self.m_ip_image_encoder : IImageEncoder = None
        
        # convert image between pil/numpy to torch tensor
        # self.m_vae_imgproc = VaeImageProcessor()
        
    @property
    def num_trained_timesteps(self) -> int:
        ''' number of timesteps the model has been trained for
        '''
        return C.MaxTimestep_sd_v15
        
    def init_diffusion_state(self, dstate : DiffusionState, 
                        reset_latent : bool = True,
                        reset_timestep : bool = True,
                        target_device : torch.device = None,
                        target_dtype : torch.dtype = None,
                        batch_size : int = 1) -> None:
        ''' initialize the diffusion state, by filling in the missing information in state given what is already there.
        '''
        import copy
        
        if target_device is None:
            target_device = self.m_unet.device
        
        # encode text
        if dstate.positive_prompt.embedding is None:
            emb = self.m_text_model.encode(dstate.positive_prompt.text).word_embedding
            dstate.positive_prompt.embedding = emb.to(device=target_device)
        
        if dstate.negative_prompt.embedding is None:
            emb = self.m_text_model.encode(dstate.negative_prompt.text).word_embedding
            dstate.negative_prompt.embedding = emb.to(device=target_device)
            
        # setup scheduler
        if dstate.scheduler is None:
            dstate.scheduler = copy.deepcopy(self.m_scheduler)
            
        if dstate.timestep_index_sequence is not None:
            dstate.set_timestep_indices(dstate.timestep_index_sequence)
            
        if reset_timestep:
            dstate.set_current_step_index(None)
            # dstate.current_step_index = None # which means the diffusion has not started yet
            
        if dstate.latent_clean is not None:
            dstate.latent_clean = dstate.latent_clean.to(device=target_device, dtype=target_dtype)
            
        if dstate.latent is not None:
            dstate.latent = dstate.latent.to(device=target_device, dtype=target_dtype)
            
        # initialize latent if necessary
        if dstate.latent is None or reset_latent:
            latent_shape = self.m_unet.get_latent_shape_by_ouptut_size(dstate.target_width, dstate.target_height)
            n,h,w,c = batch_size, *latent_shape[1:]
            dstate.latent = torch.randn((n,h,w,c), generator=dstate.random_generator)
            dstate.latent *= self.m_scheduler.scheduler.init_noise_sigma
            dstate.latent = dstate.latent.to(device=target_device, dtype=target_dtype)
            
        # process controlnet images
        for key, val in dstate.controlnet_data.items():
            assert val.image_data is not None, f'image_data must be specified for controlnet {key}'
            
            h, w = val.height, val.width
            assert h == dstate.target_height and w == dstate.target_width, \
                f'controlnet {key} image size must match target size'
            
        # process ip adapter images
        if dstate.image_prompt_data:
            ipa_helper = IPAdapterHelper.from_unet(self.m_unet, image_encoder=self.m_ip_image_encoder)
            ipa_names_expect = ipa_helper.get_adapters().keys()
            ipa_names_input = dstate.image_prompt_data.keys()
            ipa_names_missing = set(ipa_names_expect) - set(ipa_names_input)
            ipa_names_unhandled = set(ipa_names_input) - set(ipa_names_expect)
            assert ipa_names_missing == set(), f'missing ip adapter images: {ipa_names_missing}'
            assert ipa_names_unhandled == set(), f'unhandled ip adapter input: {ipa_names_unhandled}'
                    
            for key, val in dstate.image_prompt_data.items():
                # already has embedding, skip
                if val.positive_embedding is not None and val.negative_embedding is not None:
                    val.positive_embedding = val.positive_embedding.to(device=target_device, dtype=target_dtype)
                    val.negative_embedding = val.negative_embedding.to(device=target_device, dtype=target_dtype)
                else:
                    # either positive or negative embedding is missing, need to process
                    ip_enc = ipa_helper.encode_image_to_embedding({val.name: val.image})
                    if val.positive_embedding is None:
                        val.positive_embedding = ip_enc[val.name].positive.to(device=target_device, dtype=target_dtype)
                        
                    if val.negative_embedding is None:
                        val.negative_embedding = ip_enc[val.name].negative.to(device=target_device, dtype=target_dtype)
            
    def step_denoise(self, dstate_batch : list[DiffusionState], 
                     apply_latent_mask : bool = True) -> bool:        
        ''' perform denoising diffusion for one step, if a DiffusionState's batch size > 1, it will be splitted
        into batches and combine again after denoising.
        
        parameters
        ------------
        dstate_batch : list[DiffusionState]
            list of diffusion states to be denoised, the given dstates will be batched together
        apply_latent_mask : bool
            whether to denoise only the masked part of the latent, if False, then denoise the whole latent.
            The masked part is specified by dstate.latent_denoise_mask, which is a weighted mask of the same shape as latent.
        
        return
        --------
        should_continue : bool
            whether the denoising should continue, if False, then all dstate in the batch has completed
        '''        
        import copy
        idxlist_per_dstate : list[np.ndarray] = []
        num_added_dstate = 0
        
        dstate : DiffusionState = None
        dstate_list_to_process : list[DiffusionState] = []
        
        for i, dstate in enumerate(dstate_batch):
            assert dstate.timestep_index_sequence is not None, f'dstate {i} must have timestep_index_sequence'
            
            # need to process this dstate?
            if dstate.scheduler.next_step_index is None:
                idxlist_per_dstate.append(None)
                continue
            
            # need to split?
            n = dstate.latent.shape[0]
            if n == 1:  # no need to copy, just do it
                dstate_list_to_process.append(dstate)
                idxlist_per_dstate.append(np.atleast_1d(num_added_dstate))
                num_added_dstate += 1
            else:
                # split batch into individual dstate, later will combine them back
                for k in range(n):
                    sub_dstate = copy.copy(dstate)  # shallow copy, then change something
                    sub_dstate.latent = dstate.latent[k:k+1]
                    dstate_list_to_process.append(sub_dstate)
                    
                idxlist_per_dstate.append(np.arange(num_added_dstate, num_added_dstate+n))
                num_added_dstate += n
                
        if len(dstate_list_to_process) == 0:
            # nothing to process
            return False
        
        # find noise
        noise_pred = self._step_predict_noise(dstate_list_to_process)
        
        # update dstate_batch by denoising
        for i, dstate in enumerate(dstate_batch):
            idxlist = idxlist_per_dstate[i]
            if idxlist is None:
                continue
            
            _noise_pred = noise_pred[idxlist]
            self._step_apply_denoise(dstate, _noise_pred, apply_latent_mask=apply_latent_mask)
        
        return True
            
    def _step_apply_denoise(self, dstate : DiffusionState, predicted_noise : torch.Tensor, apply_latent_mask : bool = False):
        ''' apply denoise to a single diffusion state, this will update the latent and step index
        
        parameters
        ------------
        dstate : DiffusionState
            diffusion state to be updated
        predicted_noise : torch.Tensor
            predicted noise, which is the output of unet
        apply_latent_mask : bool
            whether to denoise only the masked part of the latent, if False, then denoise the whole latent.
            The masked part is specified by dstate.latent_denoise_mask, which is a weighted mask of the same shape as latent.
        '''
        
        apply_latent_mask = dstate.latent_denoise_mask is not None and apply_latent_mask
        if apply_latent_mask:
            assert dstate.latent.device == dstate.latent_denoise_mask.device, 'latent and latent_denoise_mask must be on the same device'
            assert dstate.latent.dtype == dstate.latent_denoise_mask.dtype, 'latent and latent_denoise_mask must have the same dtype'
        
        sch = dstate.scheduler
        _latent = dstate.latent.to(device=predicted_noise.device, 
                                   dtype=predicted_noise.dtype)
        
        rng = dstate.random_generator            
        _latent_update : torch.Tensor = sch.scheduler.step(predicted_noise, 
                                                           sch.next_timestep, 
                                                           _latent, 
                                                           generator=rng, 
                                                           return_dict=False)[0]
        
            
        # unify the dtype and device of latent, update and mask
        # then apply the update
        _latent_update = _latent_update.expand_as(dstate.latent).to(device=dstate.latent.device, dtype=dstate.latent.dtype)
        if apply_latent_mask:
            _blend_mask = dstate.latent_denoise_mask.to(device=dstate.latent.device, dtype=dstate.latent.dtype)
            dstate.latent = torch.lerp(dstate.latent, _latent_update, _blend_mask)
        else:
            dstate.latent = _latent_update

        dstate.current_step_index = sch.current_step_index
        
    def _check_dstate_precondition(self, dstate_batch : list[DiffusionState]):
        # check if all dstate are ready to be denoised and can be batched together
        for i, dstate in enumerate(dstate_batch):
            # all dstate must have batch_size == 1
            assert dstate.latent.shape[0] == 1, f'dstate {i} must have batch_size == 1'
            
            # all dstate must be update-able
            assert dstate.scheduler.next_timestep is not None, f'dstate {i} is not update-able'

        # consistency check
        # all dstate must have the same target size
        target_width = dstate_batch[0].target_width
        target_height = dstate_batch[0].target_height
        for dstate in dstate_batch:
            assert dstate.target_width == target_width and dstate.target_height == target_height, \
                'all dstate must have the same target size'
                
        # all dstate must have the same set of controlnet data
        controlnet_keys = set(dstate_batch[0].controlnet_data.keys())
        for dstate in dstate_batch:
            assert set(dstate.controlnet_data.keys()) == controlnet_keys, \
                'all dstate must have the same set of controlnet keys'
                
        # the pipeline must support all controlnet keys
        for key in controlnet_keys:
            assert key in self.m_controlnets, f'controlnet {key} is not defined in pipeline'
            
        # all dstate must have the same set of image prompt data
        image_prompt_keys = set(dstate_batch[0].image_prompt_data.keys())
        for dstate in dstate_batch:
            assert set(dstate.image_prompt_data.keys()) == image_prompt_keys, \
                'all dstate must have the same set of image prompt keys'

    def _step_predict_noise(self, dstate_batch : list[DiffusionState]) -> torch.Tensor:
        ''' perform denoising diffusion for one step, all given dstates must have the same target size and controlnet data,
        and the batch size of any latent must be 1.
        
        parameters
        -----------
        dstate_batch : list[DiffusionState]
            list of diffusion states to be denoised, the given dstates will be batched together.
            dstate.latent must have batch_size == 1, and must have been completely denoised (i.e. next_step_index should not be None)
            
        return
        ---------
        predicted_noise : torch.Tensor
            predicted_noise[i] is the predicted noise for dstate_batch[i], note that we assume the batch size of all dstate is 1.
        '''
        
        self._check_dstate_precondition(dstate_batch)
        
        # only process those diffusion state that has not completed
        next_timestep : list[int] = [dstate.scheduler.next_timestep for dstate in dstate_batch]

        # prepare image latents
        latent = torch.cat([dstate.latent for dstate in dstate_batch]).requires_grad_(False)
        latent_scaled = latent.clone()
        for i, dstate in enumerate(dstate_batch):
            sch = dstate.scheduler.scheduler
            t = next_timestep[i]
            latent_scaled[i] = sch.scale_model_input(latent[i], t)
        latent_model_input = torch.cat([latent_scaled]*2, dim=0)
        prompt_embed_neg_pos = torch.cat(
            [dstate.negative_prompt.embedding for dstate in dstate_batch] + \
            [dstate.positive_prompt.embedding for dstate in dstate_batch])

        # run controlnet
        down_block_res_samples : torch.FloatTensor = None
        mid_block_res_sample : torch.FloatTensor = None
        for key, cnet_info in self.m_controlnets.items():
            cnet_model, cent_weight = cnet_info.model, cnet_info.weight
            
            # batch all controlnet input of this type
            cnet_input_cond = []
            condition_data_layout = 'nchw'
            cnet_data_complete = True   # all dstate have this controlnet data ?
            for dstate in dstate_batch:
                cnet_data = dstate.controlnet_data.get(key)
                if cnet_data is None:
                    cnet_data_complete = False
                    break
                d = dstate.controlnet_data[key].image_data
                layout = dstate.controlnet_data[key].image_data_layout
                dnew = to_4d_tensor(d, input_layout=layout, output_layout=condition_data_layout)
                # cnet_input_cond.extend([dnew, dnew])    # for negative and positive prompt
                cnet_input_cond.append(dnew)
                
            if not cnet_data_complete:
                # missing some controlnet data, skip this controlnet
                continue
            
            cnet_input_cond : torch.Tensor = torch.cat(cnet_input_cond * 2) # negative and positive
            cnet_input_weight : torch.Tensor = torch.tensor([dstate.controlnet_data[key].weight for dstate in dstate_batch]*2)
            t = np.concatenate([next_timestep]*2)
            
            # compute controlnet output
            cnet_weight_final = cnet_input_weight * cent_weight
            
            if torch.abs(cnet_weight_final).max() < self.ControlnetMinimumWeight:
                # skip this controlnet
                continue
            
            res = cnet_model.forward(
                sample=latent_model_input,
                timestep=t,
                prompt_embedding=prompt_embed_neg_pos,
                condition=cnet_input_cond,
                condition_data_layout=condition_data_layout,
                conditioning_scale=cnet_input_weight * cent_weight
            )
            
            if down_block_res_samples is None:
                down_block_res_samples = res.down_block_res_samples
            else:
                down_block_res_samples += res.down_block_res_samples
                
            if mid_block_res_sample is None:
                mid_block_res_sample = res.mid_block_res_sample
            else:
                mid_block_res_sample += res.mid_block_res_sample
                
        # collect ip adapter embeds
        # len(ipa_embeds) = number of ip adapters
        # ipa_embeds[i].shape = (2N, 1, D), N=batch size, D=embedding dimension
        # ipa_embeds[i][0:N] = negative image prompt, ipa_embeds[i][N:2N] = positive image prompt
        if dstate_batch[0].image_prompt_data:
            ipa_embeds = []
            for key in sorted(dstate_batch[0].image_prompt_data.keys()):
                ipa_positive = []
                ipa_negative = []
                for dstate in dstate_batch:
                    ipa_positive.append(dstate.image_prompt_data[key].positive_embedding.reshape((1,-1)))
                    ipa_negative.append(dstate.image_prompt_data[key].negative_embedding.reshape((1,-1)))
                _embeds = torch.stack(ipa_negative + ipa_positive)
                ipa_embeds.append(_embeds)
            
            # ipa_embeds = torch.stack(_ipa_embeds).to(device=latent.device, dtype=latent.dtype)
            added_cond_kwargs = {'image_embeds': ipa_embeds}
        else:
            added_cond_kwargs = None
        
        # run unet to get noise component
        t = np.concatenate([next_timestep]*2)
        noise_pred = self.m_unet.forward(
            sample=latent_model_input,
            timestep=torch.tensor(t),
            prompt_embedding=prompt_embed_neg_pos,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # denoise
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * dstate.guidance_scale
        
        return noise_pred
    
    @torch.no_grad()
    def decode_latent_to_image(self, latent : torch.Tensor, 
                               force_upcast : bool = None,
                               output_dtype : torch.dtype = None ) -> np.ndarray:
        ''' decode latent to image
        
        parameters
        -------------
        latent : torch.Tensor
            latent to be decoded, NHWC format
        force_upcast : bool
            whether to force upcast the model to float32, if None, then use the value in config.
            Only applicable when self.m_vae_fp32 is not None.
        output_dtype : None | torch.float16 | torch.float32 | torch.uint8
            output dtype, if None, then use the same dtype as latent.
            For float image, the range is within (0,1).
            For uint8 image, the range is within (0,255).
            
        return
        --------
        image : np.ndarray
            decoded image, NHWC format
        '''
        if force_upcast is None:
            force_upcast = self.m_vae.config.get('force_upcast')
            
        torch_dtype = latent.dtype
        
        if force_upcast:
            assert self.m_vae_fp32 is not None, 'force_upcast is only usable when self.m_vae_fp32 is not None'
            vae = self.m_vae_fp32
            torch_dtype = torch.float32
        else:
            vae = self.m_vae
            
        output_dtype = latent.dtype if output_dtype is None else output_dtype
        img = vae.decode(latent.to(dtype=torch_dtype) / vae.latent_scaling_factor, 
                       output_dtype=output_dtype, 
                       output_data_layout='nhwc')
        # img = vae.image_from_decode_output(z)
        return img.detach().cpu().numpy()
    
    @torch.no_grad()
    def encode_image_to_latent(self, img : torch.Tensor,
                               input_data_layout : str,
                               force_upcast : bool = None) -> torch.Tensor:
        ''' encode image to latent space.
        
        parameters
        ---------------
        x : torch.Tensor
            image tensor in batch format, can be uint8 or float16/32. For float image, 
            the pixel value range is within [0,1], in 'nhwc' format
        input_data_layout : 'nhwc' | 'nchw'
            layout of the input data
            
        return
        ----------
        scaled_latent
            the encode output, in nchw layout, scaled by vae.latent_scaling_factor.
            If force_upcast is used, the dtype will be float32 regardless of the vae dtype.
        '''
        if force_upcast is None:
            force_upcast = self.m_vae.config.get('force_upcast')
            
        if force_upcast:
            vae = self.m_vae_fp32
        else:
            vae = self.m_vae
            
        z = vae.encode(img, input_data_layout=input_data_layout).mean
        z *= vae.latent_scaling_factor
        return z