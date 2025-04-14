import numpy as np
import torch
import unittest
from rich import print as pprint
from attrs import define, field

import igpy.common.image_processing as ip

from pyaigc.loader.LocalModelLoader import (
    LocalModelLoader, DiffusionModelBundle
)
import pyaigc.GlobalConfig as C
from pyaigc.pipeline.StableDiffusionWithControl import (
    StableDiffusionWithControl as SDPipeline,
    DiffusionState as SDState,
    ControlNetInfo,
    ControlnetData,
    TextPromptData,
    ImagePromptData
)
from pyaigc.helper.ip_adapter import IPAdapterHelper
from pyaigc.model.local.LocalIPAdapter import DiffusionIPAdapter

import pyaigc.TestHelpers as th

@define(kw_only=True, eq=False)
class InputCase:
    fn_image : str | None = field(default=None)
    positive_prompt : str | None = field(default='')
    negative_prompt : str | None = field(default='')
    noise_timestep_indices : np.ndarray | None = field()
    num_denoise_step : int = 10
    canny_weight : float = 0.0
    depth_weight : float = 0.0
    openpose_weight : float = 0.0
    batch_size : int = 1
    
    image : np.ndarray | None = field(default=None)    # image data
    canny_image : np.ndarray | None = field(default=None)  # uint8 canny image
    depth_image : np.ndarray | None = field(default=None)   # uint8 depth image
    openpose_image : np.ndarray | None = field(default=None)   # uint8 openpose image
    random_generator : torch.Generator | None = field(default=None)
    
    # ip adapter images
    # ip adapter name => image data
    ip_adapter_images : dict[str, C.ImageTensor] = field(factory=dict)
    
    def __attrs_post_init__(self):
        assert self.fn_image is not None or self.image is not None, \
            f'one of the fn_image or image must be given'
        if self.image is not None:
            # must be uint8 RGB image
            assert self.image.dtype == np.uint8 and self.image.ndim == 3 and self.image.shape[2] == 3, \
                f'image must be uint8 RGB image'
    
    @noise_timestep_indices.default
    def _default_noise_timestep_indices(self) -> np.ndarray:
        out = np.logspace(np.log10(500), np.log10(999), 40-1, dtype=int)
        # out = np.array([700]*9)
        return out
    
    @property
    def width(self) -> int:
        if self.image is not None:
            return self.image.shape[1]
        elif self.canny_image is not None:
            return self.canny_image.shape[1]
        elif self.depth_image is not None:
            return self.depth_image.shape[1]
        elif self.openpose_image is not None:   
            return self.openpose_image.shape[1]
    @property
    def height(self) -> int:
        if self.image is not None:
            return self.image.shape[0]
        elif self.canny_image is not None:
            return self.canny_image.shape[0]
        elif self.depth_image is not None:
            return self.depth_image.shape[0]
        elif self.openpose_image is not None:
            return self.openpose_image.shape[0]
    

    def crop_by(self, xmin : int, ymin : int, w : int, h : int):
        ''' crop the image and masks by the given bbox
        '''
        if self.image is not None:
            self.image = self.image[ymin:ymin+h, xmin:xmin+w, ...]
            
        if self.canny_image is not None:
            self.canny_image = self.canny_image[ymin:ymin+h, xmin:xmin+w, ...]
            
        if self.depth_image is not None:
            self.depth_image = self.depth_image[ymin:ymin+h, xmin:xmin+w, ...]
            
        if self.openpose_image is not None:
            self.openpose_image = self.openpose_image[ymin:ymin+h, xmin:xmin+w, ...]
            
    def resize_to(self, h : int, w : int, divisible_by : int = None):
        ''' resize the image and masks to the given size
        '''
        if divisible_by is not None:
            h = h // divisible_by * divisible_by
            w = w // divisible_by * divisible_by
            
        assert h>0 and w>0, 'height and width must be positive'
        
        if self.image is not None:
            self.image = ip.imresize(self.image, (h,w), interp_method='bilinear')
            
        if self.canny_image is not None:
            self.canny_image = ip.imresize(self.canny_image, (h,w), interp_method='nearest')
            
        if self.depth_image is not None:
            self.depth_image = ip.imresize(self.depth_image, (h,w), interp_method='nearest')
            
        if self.openpose_image is not None:
            self.openpose_image = ip.imresize(self.openpose_image, (h,w), interp_method='nearest')
            
    def resize_by_long_edge(self, long_edge_length : int, divisible_by : int = None):
        ''' resize the image and masks so that the longer edge matches the given length
        '''
        h, w = self.height, self.width
        scale = long_edge_length / max(h,w)
        h_new = round(h*scale)
        w_new = round(w*scale)
        
        self.resize_to(h_new, w_new, divisible_by=divisible_by)
        
    
@define(kw_only=True, eq=False)
class OutputCase:
    input_case : InputCase | None = field(default=None)
    
    dstate : SDState = field(default=None)
    final_images : np.ndarray = field(default=None)
    
@define(kw_only=True, eq=False)
class PipelineConfig:
    sd_model_key : str = field()
    with_canny_control : bool = field(default=False)
    with_depth_control : bool = field(default=False)
    with_openpose_control : bool = field(default=False)
    with_vae_upcast : bool = field(default=True)
    with_ip_image_encoder : bool = field(default=False)
    
    # name => key
    ip_adapters : dict[str, str] = field(factory=dict)
    
    def __attrs_post_init__(self):
        if self.ip_adapters:
            assert self.with_ip_image_encoder, 'ip_image_encoder must be enabled when ip_adapters is given'

@define(kw_only=True, eq=False)
class SDTester:    
    model_pipeline : SDPipeline = field(default=None)
    common_positive_prompt : str = field(default='masterpiece, high quality, very sharp, photorealistic')
    common_negative_prompt : str = field(default='bad face, blurry, low quality, gray, dull, lack of detail, naked')
    target_device : torch.device = field(default=C.DEFAULT_TORCH_DEVICE)
    target_dtype : torch.dtype = field(default=C.DEFAULT_TORCH_DTYPE)
    sd_model_key : str = field(default=None)
    
    def _create_pipeline(self) -> SDPipeline:
        return SDPipeline()
    
    def load_model_pipeline(self, config : PipelineConfig) -> tuple[SDPipeline, DiffusionModelBundle]:
        
        # load models
        model_config = C.ModelConfigs.get_instance().get_model_config(config.sd_model_key)
        if not config.with_canny_control:
            model_config.controlnets.canny = None
        if not config.with_depth_control:
            model_config.controlnets.depth = None
        if not config.with_openpose_control:
            model_config.controlnets.openpose = None
        if not config.with_ip_image_encoder:
            model_config.ip_adapter.encoder = None
            
        # load ip adapters
        if model_config.ip_adapter is not None:
            model_config.ip_adapter.rename_adapters(name2key=config.ip_adapters)
            
            # load to cpu before added to unet, to save gpu memory
            for v in model_config.ip_adapter.adapters.values():
                v.torch_device = torch.device('cpu')
                v.torch_dtype = torch.float32
            
        model_bundle = LocalModelLoader.get_instance().get_model_bundle(model_config)
            
        # construct pipeline
        obj = self._create_pipeline()
        obj.m_text_model = model_bundle.text_model
        obj.m_unet = model_bundle.unet
        
        for info, name in [(model_config.controlnets.canny, 'canny'), 
                           (model_config.controlnets.depth, 'depth'),
                           (model_config.controlnets.openpose, 'openpose')]:
            if info is not None:
                cnet_model = model_bundle.controlnets.get(info.key)
                assert cnet_model is not None, f'control net {info.key} not loaded'
                obj.m_controlnets[name] = ControlNetInfo(model=cnet_model, weight=1.0)
            
        obj.m_scheduler = model_bundle.scheduler
        obj.m_vae = model_bundle.vae
        obj.m_ip_image_encoder = model_bundle.ip_image_encoder
        
        if config.with_vae_upcast:
            import copy
            obj.m_vae_fp32 = copy.deepcopy(obj.m_vae).to(dtype=torch.float32)
        
        self.model_pipeline = obj
        self.sd_model_key = config.sd_model_key
        
        if model_bundle.ip_adapters:
            self._setup_ip_adapters(model_bundle.ip_adapters)
        
        return obj, model_bundle
    
    def _setup_ip_adapters(self, name2model : dict[str, DiffusionIPAdapter]):
        ipa_helper = IPAdapterHelper.from_unet(
            self.model_pipeline.m_unet,
            self.model_pipeline.m_ip_image_encoder,
            ip_adapters=name2model
        )
        
        # install
        ipa_helper.install_adapters(replace_existing=True)
        
        pass
    
    def _create_diffusion_state(self, target_width : int, target_height : int) -> SDState:
        dstate = SDState(
            target_height=target_height,
            target_width=target_width
        )
        return dstate
    
    def denoise_dstate(self, dstate_list : list[SDState], 
                       num_step : int = None,
                       apply_latent_mask : bool = True,
                       noise : torch.Tensor = None) -> bool:
        ''' denoise all dstates, return whether the denoise can continue
        
        parameters
        ----------
        noise : torch.Tensor
            the initial noise that have been added to the clean latent
        '''
        # denoise
        if num_step is None:
            num_step = np.inf
            
        iter = 0
        should_continue : bool = False
        pline = self.model_pipeline
        
        from tqdm.auto import tqdm
        num_steps_required = max([ds.num_remaining_denosie_step for ds in dstate_list])
        num_steps_required = min(num_steps_required, num_step)

        for it in tqdm(range(num_steps_required)):
            # update latent with blended latent
            for dstate in dstate_list:
                if dstate.scheduler.next_step_index is not None and dstate.is_inpainting and apply_latent_mask:
                    dstate.latent = dstate.get_latent_for_inpainting(latent_noise=noise)
            
            should_continue = pline.step_denoise(dstate_list, apply_latent_mask=apply_latent_mask)
            if not should_continue:
                print(f'denoise early stop at step {it}')
                break

        return should_continue
        
    def add_noise_to_dstate(self, dstate:SDState, 
                             to_timestep_index : int,
                             num_denoise_step : int = None,
                             noise : torch.Tensor | None = None,
                             prepare_for_denoise : bool = True,
                             apply_latent_mask : bool = True):
        lt = dstate.latent
        
        if noise is None:
            rng = dstate.random_generator
            noise = torch.randn(lt.shape, generator=rng).to(dtype=lt.dtype, device=lt.device)
        else:
            assert lt.shape == noise.shape, 'noise must have the same shape as latent'
            noise = noise.to(dtype=lt.dtype, device=lt.device)
            
        # ltnew = dstate.scheduler.add_noise(lt, noise, to_timestep_index)
        # dstate.latent = ltnew        
        dstate.add_masked_noise_to_latent(noise, to_timestep_index = to_timestep_index, 
                                          apply_latent_mask=apply_latent_mask)
        
        if num_denoise_step is None:
            num_denoise_step = dstate.num_all_timesteps
        
        # prepare to denoise
        if prepare_for_denoise:
            self.model_pipeline.init_diffusion_state(dstate, 
                                                    reset_latent=False, 
                                                    reset_timestep=True)
            dstate.set_timestep_sequence_by_num_steps(num_denoise_step, max_timestep_index=to_timestep_index)
    
    def init_input_case(self, input_case: InputCase,
                        auto_resize_div8 : bool = False,
                        auto_resize_long_edge_length : int = None):
        ''' initialize input case by loading images from files, and resize them
        '''
        if input_case.image is None:
            # load image and resize it
            simg = th.SampleImage.from_file(input_case.fn_image)
            if auto_resize_div8:
                if auto_resize_long_edge_length is not None:
                    long_edge = auto_resize_long_edge_length
                else:
                    long_edge = max(simg.image.shape[:2]) // 8 * 8
                simg = simg.resize_by_long_edge(long_edge, divisible_by=8)
            else:
                if auto_resize_long_edge_length is not None:
                    simg = simg.resize_by_long_edge(auto_resize_long_edge_length)
            input_case.image = simg.image
        
        if input_case.canny_image is None and input_case.canny_weight > 0:
            simg = th.SampleImage.from_ndarray(input_case.image)
            input_case.canny_image = simg.get_canny_edge()
        
    def init_output_case(self, input_case : InputCase) -> OutputCase:
        # init output case
        output_case = OutputCase()
        output_case.input_case = input_case
        h,w = input_case.image.shape[:2]
        
        assert h % 8 == 0 and w % 8 == 0, 'image size must be divisible by 8'
        
        dstate = self._create_diffusion_state(target_width=w, target_height=h)
        output_case.dstate = dstate
        
        # append common prompts
        if input_case.positive_prompt is not None and len(input_case.positive_prompt) > 0:
            dstate.positive_prompt.text = input_case.positive_prompt + ',' + self.common_positive_prompt
        else:
            dstate.positive_prompt.text = self.common_positive_prompt
            
        if input_case.negative_prompt is not None and len(input_case.negative_prompt) > 0:
            dstate.negative_prompt.text = input_case.negative_prompt + ',' + self.common_negative_prompt
        else:
            dstate.negative_prompt.text = self.common_negative_prompt
        
        # setup control net
        if input_case.canny_weight > 0 and input_case.canny_image is not None:
            dstate.controlnet_data['canny'] = ControlnetData(
                image_data=input_case.canny_image[None,...,None],
                image_data_layout='nhwc',
                weight=input_case.canny_weight
            )
            
        if input_case.openpose_weight > 0 and input_case.openpose_image is not None:
            # openpose image is 3 channel uint8 image
            dstate.controlnet_data['openpose'] = ControlnetData(
                image_data=input_case.openpose_image[None,...],
                image_data_layout='nhwc',
                weight=input_case.openpose_weight
            )
            
        if input_case.depth_weight > 0 and input_case.depth_image is not None:
            depth_image = input_case.depth_image
            # depth image is always single channel, if not, convert to single channel
            if depth_image.ndim >= 3 and depth_image.shape[2] > 1:
                depth_image = depth_image[..., 0]
            dstate.controlnet_data['depth'] = ControlnetData(
                image_data=depth_image[None,...],
                image_data_layout='nhwc',
                weight=input_case.depth_weight
            )
            
        # fill others
        dstate.rng = input_case.random_generator
        
        # setup image prompt data
        if input_case.ip_adapter_images:
            for name, ipa_image in input_case.ip_adapter_images.items():
                dstate.image_prompt_data[name] = ImagePromptData(
                    name = name,
                    image = ipa_image
                )
        
        # init the diffusion state
        self.model_pipeline.init_diffusion_state(dstate, 
                                                reset_latent=True, 
                                                reset_timestep=True,
                                                target_device=self.target_device,
                                                target_dtype=self.target_dtype,
                                                batch_size=input_case.batch_size)
        dstate.set_timestep_sequence_by_num_steps(input_case.num_denoise_step)
        
        return output_case