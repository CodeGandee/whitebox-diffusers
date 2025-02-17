# helpers for testing
import torch
import numpy as np
import cv2
import PIL.Image
from typing import Union
import igpy.common.inout as cio
from attrs import define, field
import attrs.validators as _av

from diffusers import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionPipeline
)

import pyaigc.model.local.LocalModel as lm
from pyaigc.model.DiffusionScheduler import (
    DiffusionScheduler,
    IDiffusionScheduler,
    DiffusionScheduler_EulerDescrete,
    SchedulerTypeNames
)

from pyaigc.model import *
import pyaigc.GlobalConfig as C
from pyaigc.loader.LocalModelLoader import LocalModelLoader

# hack: this is a workaround for the fact that torchscript doesn't support non tensor arguments
class MyUNet2D(UNet2DConditionModel):
    def __init__(self, unet_model : UNet2DConditionModel) -> None:
        super().__init__()
        self.__dict__.update(unet_model.__dict__)
        
    def forward(self, sample, timestep, encoder_hidden_states) -> torch.Tensor:
        return super().forward(sample, timestep, encoder_hidden_states, return_dict=False)

class FakeInput_Unet2DConditionModel:
    def __init__(self) -> None:
        self.m_sample : torch.Tensor = torch.randn(2,4,64,64, dtype=torch.float32)
        self.m_timesteps : torch.Tensor = torch.tensor(np.random.randint(0, 100, (2,)), dtype=torch.int32)
        self.m_hidden_states : torch.Tensor = torch.randn(2,77,768, dtype=torch.float32)
    

class SampleImage:
    class ResizeMethod:
        Crop = 'crop'
        Scale = 'scale'
        
    def __init__(self, img : np.ndarray = None) -> None:
        self.m_image : np.ndarray = img
        
    @classmethod
    def from_file(cls, fn : str) -> 'SampleImage':
        img = cio.imread(fn)
        return cls(img)
    
    @classmethod
    def from_ndarray(cls, img : np.ndarray) -> 'SampleImage':
        return cls(img)
    
    @property
    def image(self) -> np.ndarray:
        return self.m_image
    
    @property
    def width(self) -> int:
        return self.m_image.shape[1]
    
    @property
    def height(self) -> int:
        return self.m_image.shape[0]
    
    def clone(self) -> 'SampleImage':
        return SampleImage.from_ndarray(self.m_image.copy())
    
    def resize_by_long_edge(self, long_edge : int, 
                            divisible_by : int = None,
                            method : str = ResizeMethod.Scale) -> 'SampleImage':
        ''' resize the image so that the longer edge is long_edge, and divisible by divisible_by,
        using crop or scale.
        '''
        
        # resize by scale
        new_image = self.m_image
        if method == self.ResizeMethod.Scale:
            height, width = self.m_image.shape[:2]
            if height > width:
                new_height = long_edge
                new_width = int(width * long_edge / height)
            else:
                new_width = long_edge
                new_height = int(height * long_edge / width)
            new_image = cv2.resize(self.m_image, (new_width, new_height))
            
        # ensure divisible by, using crop
        if divisible_by is not None:
            height, width = new_image.shape[:2]
            new_height = height - height % divisible_by
            new_width = width - width % divisible_by
            new_image = new_image[:new_height, :new_width]
            
        self.m_image = new_image
        return self
    
    def as_uint8(self) -> 'SampleImage':
        # float in (0,1) to uint8 in (0,255)
        if self.m_image.dtype in (np.float32, np.float64, np.float16):
            self.m_image = (self.m_image * 255).astype(np.uint8)
        return self
    
    def as_float32(self) -> 'SampleImage':
        # uint8 in (0,255) to float in (0,1)
        if self.m_image.dtype == np.uint8:
            self.m_image = self.m_image.astype(np.float32) / 255
        return self
    
    def get_image(self, output_class : Union[np.ndarray, PIL.Image.Image] = np.ndarray) -> Union[np.ndarray, PIL.Image.Image]:
        if output_class == np.ndarray:
            return self.m_image
        elif output_class == PIL.Image.Image:
            return PIL.Image.fromarray(self.m_image)
        else:
            raise ValueError('unsupported output class')
        
    def get_canny_edge(self, output_class : Union[np.ndarray, PIL.Image.Image] = np.ndarray,
                       thres_1 : float = 100.0, thres_2 : float = 200.0) -> Union[np.ndarray, PIL.Image.Image]:
        img = self.get_image(np.ndarray)
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])
        img = cv2.Canny(img, thres_1, thres_2)
        
        if output_class == np.ndarray:
            return img
        elif output_class == PIL.Image.Image:
            return PIL.Image.fromarray(img)
        else:
            raise ValueError('unsupported output class')
    
@define(kw_only=True)
class DiffusionPipelineComponent:
    ''' models required to construct diffusion pipeline
    '''
    sd_model_key : str
    
    text_model : IDiffusionTextModel | None = field(default=None, validator=_av.optional(_av.instance_of(IDiffusionTextModel)))
    
    # required by sdxl pipeline
    text_model_2 : IDiffusionTextModel | None = field(default=None, validator=_av.optional(_av.instance_of(IDiffusionTextModel)))
    
    unet : IDiffusionUnet | None = field(default=None, validator=_av.optional(_av.instance_of(IDiffusionUnet)))
    vae : IDiffusionVae | None = field(default=None, validator=_av.optional(_av.instance_of(IDiffusionVae)))
    scheduler : IDiffusionScheduler | None = field(default=None, validator=_av.optional(_av.instance_of(IDiffusionScheduler)))
    
    controlnet_canny : IControlnet | None = field(default=None, validator=_av.optional(_av.instance_of(IControlnet)))
    controlnet_depth : IControlnet | None = field(default=None, validator=_av.optional(_av.instance_of(IControlnet)))
    
    @classmethod
    def from_sdxl_model(cls, sdxl_path : str, 
                        dtype : torch.dtype,
                        device : torch.device,
                        controlnet_canny_path : str = None,
                        cotnrolnet_depth_path : str = None,
                        scheduler_type : str = None) -> 'DiffusionPipelineComponent':
        raise NotImplementedError('not implemented')
    
        logf = C.Logme.get_instance().info
        
        logf(f'loading sdxl model from {sdxl_path}')
        pline_base : StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(sdxl_path).to(device=device, dtype=dtype)
        
        canny_model : lm.DiffusionControlnet = None
        if controlnet_canny_path is not None:
            logf(f'loading canny controlnet from {controlnet_canny_path}')
            _cnet_model = ControlNetModel.from_pretrained(controlnet_canny_path).to(device=device, dtype=dtype)
            canny_model = lm.DiffusionControlnet.from_model(_cnet_model)
            
        depth_model : lm.DiffusionControlnet = None
        if cotnrolnet_depth_path is not None:
            logf(f'loading depth controlnet from {cotnrolnet_depth_path}')
            _cnet_model = ControlNetModel.from_pretrained(cotnrolnet_depth_path).to(device=device, dtype=dtype)
            depth_model = lm.DiffusionControlnet.from_model(_cnet_model)
            
        text_model = lm.DiffusionTextModel.from_clip(
            tokenizer=pline_base.tokenizer,
            encoder_model=pline_base.text_encoder
        )
        
        text_model_2 = lm.DiffusionTextModel.from_clip(
            tokenizer=pline_base.tokenizer_2,
            encoder_model=pline_base.text_encoder_2
        )
        
        vae_model = lm.DiffusionVae.from_model(
            model = pline_base.vae
        )
        
        unet_model = lm.DiffusionUnet.from_diffuser(
            unet = pline_base.unet
        )
        
        scheduler = DiffusionScheduler_EulerDescrete.from_diffuser_scheduler(pline_base.scheduler)
        # if scheduler_type is None:
        #     scheduler = DiffusionScheduler.from_diffuser_scheduler(pline_base.scheduler)
        # else:
        #     scheduler = DiffusionScheduler.from_config(
        #         config = pline_base.scheduler.config,
        #         typename = SchedulerTypeNames.DPM_2M
        #     )
        
        logf(f'finished loading sdxl model from {sdxl_path}')
            
        return cls(
            sd_model_key = 'sdxl',
            text_model = text_model,
            text_model_2 = text_model_2,
            unet = unet_model,
            vae = vae_model,
            scheduler = scheduler,
            controlnet_canny = canny_model,
            controlnet_depth = depth_model
        )
    
    @classmethod
    def from_sd_model_key(cls, sd_model_key : str, 
                          with_text_model : bool = True,
                          with_unet : bool = True,
                          with_vae : bool = True,
                          with_canny_control : bool = False,
                          with_depth_control : bool = False) -> 'DiffusionPipelineComponent':
        raise NotImplementedError('not implemented')
    
        dao = LocalModelLoader.get_instance()
        
        if with_text_model:
            C.log_info(f'loading models for sd model {sd_model_key}')
            key_enc, key_tok = C.ModelConfigs.key_of_text_model_by_sd_model(sd_model_key)
            text_model = dao.get_text_model(model_key=key_enc, tokenizer_key=key_tok)
        else:
            text_model = None
        
        if with_unet:
            C.log_info(f'loading unet for sd model {sd_model_key}')
            key = C.ModelConfigs.key_of_unet_by_sd_model(sd_model_key)
            unet = dao.get_diffusion_unet(key)
        else:
            unet = None
        
        if with_vae:
            C.log_info(f'loading vae for sd model {sd_model_key}')
            key = C.ModelConfigs.key_of_vae_by_sd_model(sd_model_key)
            vae = dao.get_vae(key)
        else:
            vae = None
        
        C.log_info(f'loading scheduler for sd model {sd_model_key}')
        key = C.ModelConfigs.key_of_scheduler_by_sd_model(sd_model_key)
        scheduler = dao.get_scheduler(key)
        # mutable_config = dict(_scheduler.config)
        # mutable_config['use_karras_sigmas'] = False
        # scheduler = DiffusionScheduler_EulerDescrete.from_config(mutable_config)
        
        C.log_info(f'loading controlnets for sd model {sd_model_key}')
        controlnet_canny = None
        if with_canny_control:
            key = C.ControlNetModelKeys.Canny_sd_v15
            controlnet_canny = dao.get_controlnet(key)
        
        controlnet_depth = None
        if with_depth_control:
            key = C.ControlNetModelKeys.Depth_sd_v15
            controlnet_depth = dao.get_controlnet(key)
        
        C.log_info(f'finished loading models for sd model {sd_model_key}')
        return cls(sd_model_key=sd_model_key,
                   text_model=text_model,
                   unet=unet,
                   vae=vae,
                   scheduler=scheduler,
                   controlnet_canny=controlnet_canny,
                   controlnet_depth=controlnet_depth)
    