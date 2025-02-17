# for loading models related to stable diffusion
import torch
import json
from weakref import WeakValueDictionary
from typing import Any, Callable
from attrs import define, field

from igpy.ml.util_diffusers import load_safetensor_as_torch_tensor

from diffusers.models import UNet2DConditionModel, ControlNetModel, AutoencoderKL, ModelMixin
from diffusers.schedulers import (
    UniPCMultistepScheduler, SchedulerMixin,
    DPMSolverMultistepScheduler)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from transformers.image_processing_utils import BaseImageProcessor

import pyaigc.model.local.LocalModel as lm
import pyaigc.model.local.LocalImageEncoder as lm_enc
import pyaigc.model.local.LocalIPAdapter as lm_ipa

from pyaigc.model.DiffusionScheduler import (
    DiffusionScheduler,
    DiffusionScheduler_EulerDescrete
)
import pyaigc.GlobalConfig as C
from pyaigc.GlobalConfig import (
    SDModelConfig, ModelComponent, 
    TextModelComponent, SchedulerComponent
)
from pyaigc.loader.utils import DiffusionModelBundle

logf = C.Logme.get_instance().debug

CacheKey = tuple[str, torch.device]  # (model_key, torch_device)

@define(kw_only=True, slots=False)
class CacheConfig:
    # by default, unet is not cached
    cache_unet : bool = field(default=False)
    
    cache_vae : bool = field(default=True)
    cache_controlnet : bool = field(default=True)
    cache_ip_adapter : bool = field(default=True)
    cache_ip_image_encoder : bool = field(default=True)
    cache_text_model : bool = field(default=True)

class LocalModelLoader:
    
    # singleton instance, you can also create your own instance
    _instance : 'LocalModelLoader' = None
    
    def __init__(self) -> None:
        self.m_model_cache : WeakValueDictionary[CacheKey, Any] = WeakValueDictionary()
        self.m_cache_config : CacheConfig = CacheConfig()
        
    @classmethod
    def get_instance(cls) -> 'LocalModelLoader':
        if cls._instance is None:
            cls._instance = LocalModelLoader()
        return cls._instance
        
    @classmethod
    def make_cache_key(cls, model_key:str, torch_device : C.TorchDevice):
        dev = C.get_canonical_torch_device(torch_device)
        return (model_key, dev)
    
    def clear_cache(self) -> None:
        ''' remove all models from cache
        '''
        self.m_model_cache.clear()
    
    @classmethod
    def _resolve_dtype_and_device(cls, torch_dtype : torch.dtype, torch_device : C.TorchDevice) -> tuple[torch.dtype, torch.device]:
        if torch_dtype is None:
            torch_dtype = C.DEFAULT_TORCH_DTYPE
        if torch_device is None:
            torch_device = C.DEFAULT_TORCH_DEVICE
            
        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        
        return torch_dtype, torch_device
        
    def get_model_in_cache(self, model_key : str, torch_device : C.TorchDevice = None) -> ModelMixin:
        ''' get a model from cache (including recycle bin), if not found, return None
        '''
        if torch_device is None:
            devlist = C.get_all_torch_devices()
        else:
            devlist = [torch_device]
        
        cache = self.m_model_cache
        for dev in devlist:
            ckey = self.make_cache_key(model_key, dev)
            model = cache.get(ckey)
            if model is not None:
                return model
            
    @classmethod
    def _resolve_component_dtype_and_device(cls, md: ModelComponent, 
                                            dtype_fallback : torch.dtype = None, 
                                            device_fallback : C.TorchDevice = None) -> tuple[torch.dtype, torch.device]:
        if md.torch_dtype is not None:
            dtype = md.torch_dtype
        else:
            dtype = dtype_fallback
            
        if md.torch_device is not None:
            device = md.torch_device
        else:
            device = device_fallback
        
        # if None, use system default
        dtype, device = cls._resolve_dtype_and_device(dtype, device)
        
        return dtype, device
        
    
    def get_model_bundle(self, model_config : SDModelConfig,
                        torch_dtype : torch.dtype = None,
                        torch_device : C.TorchDevice = None) -> DiffusionModelBundle:
        ''' get all models given by a model config. If dtype and device are not specified,
        use the dtype and device specified in the model config, if those are not specified either,
        use the default dtype and device specified in GlobalConfig
        
        parameters
        --------------
        model_config : SDModelConfig
            model config which designates the model components to be loaded
        torch_dtype : torch.dtype
            dtype of the loaded models, if given, will override the dtype specified in the model config
        torch_device : str | torch.device
            device of the loaded models, if given, will override the device specified in the model config
            
        return
        -------
        out : dict[str, Any]
            a dictionary of loaded models, with keys as the model component keys
        '''
        assert model_config.is_valid(), f'invalid model config {model_config}, some components are missing or invalid'
        
        import copy
        out = DiffusionModelBundle()
        mc = copy.deepcopy(model_config)
        sd_version = mc.get_sd_version()
        
        if torch_dtype is not None:
            mc.set_torch_dtype(torch_dtype)
        if torch_device is not None:
            mc.set_torch_device(torch_device)
        
        # load scheduler
        if mc.scheduler is not None and mc.scheduler.path is not None:
            sch = self.get_scheduler(mc.scheduler)
            out.scheduler = sch
            
        # load text models
        for i, tm_config in enumerate((mc.text_model, mc.text_model_2)):
            if tm_config is None or tm_config.encoder.path is None or tm_config.tokenizer.path is None:
                continue
            
            text_model = self.get_text_model(tm_config)
            if i == 0:
                out.text_model = text_model
            elif i == 1:
                out.text_model_2 = text_model
                
        # load unet
        if mc.unet is not None and mc.unet.path is not None:
            unet = self.get_diffusion_unet(mc.unet)
            out.unet = unet
            
        # load vae
        if mc.vae is not None and mc.vae.path is not None:
            vae = self.get_vae(mc.vae)
            out.vae = vae
        
        # load controlnets
        if mc.controlnets is not None:
            for cnet_config in (mc.controlnets.canny, mc.controlnets.depth):
                if cnet_config is None or cnet_config.path is None:
                    continue
                
                cnet = self.get_controlnet(cnet_config)
                out.controlnets[cnet_config.key] = cnet
                
        # load adapters
        if mc.ip_adapter is not None:
            if mc.ip_adapter.encoder is not None and mc.ip_adapter.encoder.path is not None:
                img_encoder = self.get_ip_image_encoder(mc.ip_adapter.encoder)
                out.ip_image_encoder = img_encoder
            
            if mc.ip_adapter.adapters is not None:
                for ip_name, ip_config in mc.ip_adapter.adapters.items():
                    if ip_config.path is not None:
                        ip_adapter = self.get_ip_adapter(ip_config, sd_version=sd_version, name = ip_name)
                        out.ip_adapters[ip_name] = ip_adapter
                
        return out
        
    def get_diffusion_unet(self, model_info : ModelComponent, use_cache : bool = None) -> lm.DiffusionUnet:
        ''' get or load a stable diffusion unet model
        
        parameters
        ------------
        model_info : ModelComponent
            model component config
            
        return
        --------
        model : UNet2DConditionModel
            the loaded model
        '''
        if use_cache is None:
            use_cache = self.m_cache_config.cache_unet
        model : UNet2DConditionModel = self._load_model_component(
            model_info, UNet2DConditionModel, use_cache=use_cache)
        out = lm.DiffusionUnet.from_diffuser(model, model_key=model_info.key)
        
        if use_cache:
            out.set_mutable(False)
            
        return out
    
    def get_ip_adapter(self, 
                       model_info : ModelComponent, 
                       sd_version : str,
                       name : str = None,
                       use_cache:bool = None) -> lm_ipa.DiffusionIPAdapter:
        ''' get or load an ip adapter model
        
        parameters
        ------------
        model_info : ModelComponent
            model component config
        sd_version : str
            stable diffusion model version, see GlobalConfig.SDVersion
        name : str
            name of the ip adapter, used to identify this adapter in diffusion model. 
            If None, use the key in model_info
            
        return
        --------
        model : DiffusionIPAdapter
            the loaded model
        '''
        
        def func_construct_model(info : ModelComponent) -> lm_ipa.DiffusionIPAdapter:
            info.path = model_info.path
            dtype = info.torch_dtype
            device = info.torch_device
            
            # check extension, '.bin' or '.safetensor'
            ipa_data : dict[str, dict[str, torch.Tensor]] = None
            if info.path.endswith('.bin'):
                ipa_data = torch.load(info.path, map_location='cpu')
                metadata = None
            elif info.path.endswith('.safetensor'):
                sfd = load_safetensor_as_torch_tensor(info.path)
                ipa_data = sfd.data
                metadata = sfd.metadata
                
            assert set(ipa_data.keys()) == set(('image_proj', 'ip_adapter')), f'ip_adapter data keys should be (image_proj, ip_adapter) but got {ipa_data.keys()}'
            
            for key, val in ipa_data.get('ip_adapter').items():
                ipa_data['ip_adapter'][key] = val.to(dtype=dtype, device=device)
            for key, val in ipa_data.get('image_proj').items():
                ipa_data['image_proj'][key] = val.to(dtype=dtype, device=device)
                
            _name = name or info.key
            model = lm_ipa.DiffusionIPAdapter(
                model_key = info.key,
                name = _name,
                sd_version = sd_version,
                state_dict = ipa_data,    
                metadata = metadata
            )
            return model
        
        if use_cache is None:
            use_cache = self.m_cache_config.cache_ip_adapter
            
        out : lm_ipa.DiffusionIPAdapter = self._load_model_component_ex(model_info, func_construct_model, use_cache=use_cache)
        out = out.shallow_copy()
        if name is not None:
            out.set_name(name)
        
        if use_cache:
            out.set_mutable(False)
            
        return out
    
    def get_controlnet(self, model_info : ModelComponent, use_cache : bool=None) -> lm.DiffusionControlnet:
        ''' get or load a controlnet model
        
        parameters
        ------------
        model_info : ModelComponent
            model component config
        
        return
        --------
        out : DiffusionControlnet
            the loaded model
        '''
        if use_cache is None:
            use_cache = self.m_cache_config.cache_controlnet
        model : ControlNetModel = self._load_model_component(model_info, ControlNetModel, use_cache=use_cache)
        out = lm.DiffusionControlnet.from_model(model, model_key=model_info.key)
        
        if use_cache:
            out.set_mutable(False)
        return out
    
    def get_vae(self, model_info : ModelComponent, use_cache : bool = None) -> lm.DiffusionVae:
        ''' get or load a vae model
        
        parameters
        ------------
        model_info : ModelComponent
            model component config
            
        return
        --------
        model : DiffusionVae
            the loaded model
        '''
        if use_cache is None:
            use_cache = self.m_cache_config.cache_vae
            
        model : AutoencoderKL = self._load_model_component(model_info, AutoencoderKL, use_cache=use_cache)
        out = lm.DiffusionVae.from_model(model, model_key=model_info.key)
        
        if use_cache:
            out.set_mutable(False)
        return out
        
    def get_text_model(self, model_info : TextModelComponent, use_cache:bool=None) -> lm.DiffusionTextModel:
        ''' get or load a text model '''
        
        # create tokenizer
        tkpath = model_info.tokenizer.path
        assert tkpath is not None, f'invalid tokenizer path {tkpath}'
        tokenizer = CLIPTokenizer.from_pretrained(tkpath)
        
        # load encoder
        if use_cache is None:
            use_cache = self.m_cache_config.cache_text_model
        model : CLIPTextModel = self._load_model_component(model_info.encoder, CLIPTextModel, use_cache=use_cache)
        model_key = model_info.key
        out = lm.DiffusionTextModel.from_clip(tokenizer=tokenizer, encoder_model=model, model_key=model_key)
        
        if use_cache:
            out.set_mutable(False)
        return out

    def get_image_preprocessor(self, model_info : ModelComponent) -> BaseImageProcessor:
        ''' get an image preprocessor
        '''
        imgproc_path = model_info.path
        if C.ImageProcessorKeys.CLIP_ImageProcessor in model_info.key:
            imgproc = CLIPImageProcessor.from_pretrained(imgproc_path)
            return imgproc
        
    def get_scheduler(self, model_info : SchedulerComponent) -> DiffusionScheduler | DiffusionScheduler_EulerDescrete:
        ''' get a scheduler given model key and scheduler type
        '''
        scheduler_path = model_info.path
        with open(scheduler_path, 'r') as f:
            config : dict = json.load(f)
        
        scheduler_type = model_info.scheduler_type
        if scheduler_type in (C.SchedulerTypes.DPM_2M_Karras, 
                              C.SchedulerTypes.DPM_2M_SDE,
                              C.SchedulerTypes.DPM_2M_SDE_Karras,
                              C.SchedulerTypes.DPM_2M):
            return DiffusionScheduler.from_config(config, typename=scheduler_type)
        elif scheduler_type in (C.SchedulerTypes.EulerDiscrete,):
            return DiffusionScheduler_EulerDescrete.from_config(config)
        else:
            assert False, f'unsupported scheduler type {scheduler_type}'
            
    def get_ip_image_encoder(self, model_info : ModelComponent, use_cache:bool=None) -> lm_enc.ImageEncoderByCLIP:
        ''' get an image encoder for ip adapter
        '''
        if use_cache is None:
            use_cache = self.m_cache_config.cache_ip_image_encoder
            
        model : lm_enc.CLIPVisionModelWithProjection = self._load_model_component(
            model_info, lm_enc.CLIPVisionModelWithProjection, use_cache=use_cache)
        out = lm_enc.ImageEncoderByCLIP.from_encoder_model(model)
        
        if use_cache:
            out.set_mutable(False)
            
        return out
    
    def _load_model_component(self, component_config : ModelComponent, 
                              model_class : ModelMixin, 
                              use_cache : bool) -> ModelMixin:
        ''' load a model component given its config, from the path given in the config.
        
        parameters
        ------------
        component_config : ModelComponent
            model component config, which includes the key and path of the model.
        model_class : ModelMixin
            class of the model to be loaded, such as UNet2DConditionModel, ControlNetModel, AutoencoderKL
        use_cache : bool
            whether to use cache to search and store the loaded model
            
        return
        --------
        model : ModelMixin
            the loaded model
        '''
        
        assert not isinstance(component_config, TextModelComponent), f'for text model, please use the encoder component'
        assert not isinstance(component_config, SchedulerComponent), f'{component_config} not supported'
        
        model_constructor = lambda x: model_class.from_pretrained(x.path).to(device=x.torch_device, dtype=x.torch_dtype)
        out = self._load_model_component_ex(component_config, model_constructor, use_cache)
        return out
    
    def _load_model_component_ex(self, component_config : ModelComponent, 
                            model_constructor : Callable[[ModelComponent], Any],
                            use_cache : bool) -> Any:
        ''' load a model component given its config, from the path given in the config.
        
        parameters
        ------------
        component_config : ModelComponent
            model component config, which includes the key and path of the model.
        model_constructor : Any
            a function that constructs a model given the (final) component info, 
            which may be different from component_config due to hardware constraints.
        use_cache : bool
            whether to use cache to search and store the loaded model
            
        return
        --------
        model : Any
            the loaded and constructed model
        '''
    
        assert not isinstance(component_config, TextModelComponent), f'for text model, please use the encoder component'
        assert not isinstance(component_config, SchedulerComponent), f'{component_config} not supported'
        
        torch_dtype, torch_device = self._resolve_component_dtype_and_device(component_config)
        model_key = component_config.key
        model_path = component_config.path
        logf(f'loading model {model_key} from {model_path}')
        
        assert model_path is not None, f'path not found for model key {model_key}'
        
        # check if model is already loaded
        if use_cache:
            ckey = self.make_cache_key(model_key, torch_device)
            model = self.m_model_cache.get(ckey)
            if model is not None:
                logf(f'model {model_key} got from cache')
                return model
            logf(f'model {model_key} not found in cache')
            
        # load model from disk
        logf(f'loading model {model_key} from disk')
        new_info = ModelComponent(
            key=model_key,
            path=model_path,
            torch_dtype=torch_dtype,
            torch_device=torch_device
        )
        model = model_constructor(new_info)
        if use_cache:
            logf(f'caching model {model_key}')
            self.m_model_cache[ckey] = model
        
        logf(f'model loaded')
        return model
    