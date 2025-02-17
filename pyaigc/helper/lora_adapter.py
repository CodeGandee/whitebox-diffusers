import numpy as np
import torch
from attrs import define, field
from typing import Self

from pyaigc.model.local.LocalLoraAdapter import DiffusionLoraAdapter
from pyaigc.model.local.LocalModel import DiffusionUnet
import pyaigc.GlobalConfig as C

# decorator to check mutability
def check_mutable(func):
    ''' decorator to check if the model is mutable, raise error if not.
    '''
    def wrapper(self : 'LoraAdapterHelper', *args, **kwargs):
        model = self.m_unet
        if model is not None and not model.get_mutable():
            raise ValueError(f'The {model.__class__.__name__} object is not mutable, if you did not set it, it is likely that the model is shared by others, please clone() it first.')
        return func(self, *args, **kwargs)
    
    return wrapper

class LoraAdapterHelper:
    ''' install and manage lora adapter inside diffusion model
    '''
    class DataKeys:
        adapters = 'adapters'
    
    def __init__(self) -> None:
        self.m_unet: DiffusionUnet = None
        self.m_adapters : dict[str, DiffusionLoraAdapter] = {}

    @classmethod
    def from_unet(cls, unet : DiffusionUnet, 
                  adapters : list[DiffusionLoraAdapter] = None) -> Self:
        ''' create an instance from unet
        '''
        helper = cls()
        helper.m_unet = unet
        
        if adapters is not None:
            helper.m_adapters = {adapter.name: adapter for adapter in adapters}
        else:
            # read from unet
            _data = helper._retrieve_internal_data()
            if _data is not None:
                helper.m_adapters = _data.get(cls.DataKeys.adapters, {})
        
        return helper
        
    def _add_adapters_to_model(self) -> None:
        ''' add adapters to the unet model
        '''
        
        import copy
        from peft.tuners.lora.config import LoraConfig
        
        sd_version = self.m_unet.sd_version
        
        # prerequisites check
        for name, adapter in self.m_adapters.items():
            # check if adapter is compatible with unet
            is_sd_version_consistent = sd_version == adapter.sd_version
            assert is_sd_version_consistent, f'adapter sd_version {adapter.sd_version} does not match unet sd_version {sd_version}'
            
            # check if adapter has data
            assert adapter.get_data() is not None and len(adapter.get_data()) > 0, f'adapter {name} has no data'
            
        # install adapters
        if sd_version == C.SDVersion.v15:
            from diffusers.loaders.lora import LoraLoaderMixin as lora_loader
        elif sd_version == C.SDVersion.sdxl:
            from diffusers.loaders.lora import StableDiffusionXLLoraLoaderMixin as lora_loader
            
        _unet = self.m_unet.get_internal_model()
        for name, adapter in self.m_adapters.items():
            # copy tensors to avoid modification by lora_state_dict()
            _src_data = copy.copy(adapter.get_data())
            state_dict, network_alphas = lora_loader.lora_state_dict(_src_data, 
                                                                     unet_config=_unet.config)
            lora_loader.load_lora_into_unet(
                state_dict=state_dict,
                network_alphas=network_alphas,
                unet=_unet,
                adapter_name=name
            )
            
        # make all adapters effective
        fake_scales = {name: 1.0 for name in self.m_adapters.keys()}
        self._set_adapter_scale(fake_scales)
        
        # ensure dtype and device remains unchanged
        p : torch.Tensor = next(_unet.parameters())
        torch_device, torch_dtype = p.device, p.dtype
        _unet.to(device=torch_device, dtype=torch_dtype)
        
    def _set_adapter_scale(self, adapter_scales: dict[str, float]) -> None:
        ''' set adapter scales into the unet model.
        Note that you should call this function after installing adapters at least once to make them effective.
        
        parameters
        ------------
        adapter_scales : dict[str, float]
            dictionary of adapter names and their scales.
        '''
        _unet = self.m_unet.get_internal_model()
        assert set(adapter_scales.keys()).issubset(set(self.m_adapters.keys())), f'adapter scales {adapter_scales.keys()} do not match installed adapters {self.m_adapters.keys()}'
        
        ap_names : list[str] = list(adapter_scales.keys())
        ap_weights : list[float] = list(adapter_scales.values())
                
        _unet.set_adapters(ap_names, ap_weights)
        
    def _unload_adapters_from_model(self, names : list[str]) -> None:
        ''' unload given adapters from unet, repeated calls will have no effect
        '''
        
        # check if ip adapters are loaded, if yes, unload them
        _unet = self.m_unet.get_internal_model()
        if self.is_adapter_loaded():
            _unet.delete_adapters(names)
        
    def is_adapter_loaded(self) -> bool:
        ''' check if adapter is loaded
        '''
        if self.m_unet is None:
            return False
        
        has_internal_data = self._retrieve_internal_data() is not None
        return has_internal_data
        
    @check_mutable
    def install_adapters(self, clear_cache : bool = False,
                         replace_existing : bool = False) -> None:
        ''' add adapters to unet, activate them
        
        parameters
        ------------
        clear_cache : bool
            whether to clear cache after installation. If True, you can only install once because
            the data cached in adapter objects will be removed once installed.
        replace_existing : bool
            whether to replace existing adapters if they are already installed. 
            Default is False, and will raise an error if there are existing adapters.
        '''
        assert self.m_unet.get_mutable(), 'unet model is not mutable, \
            you may need to clone the unet if mutability is not controlled by you'
        
        if self.is_adapter_loaded():
            if not replace_existing:
                raise ValueError('Adapter is already installed, please remove it first')
            else:
                self.remove_adapters()
        
        # check if ip adapters have data inside them
        for adapter in self.m_adapters.values():
            if not adapter.get_data():
                raise ValueError(f'No data found for {adapter.name} in adapter. Please load data first.')
        
        self._add_adapters_to_model()
        self._register_internal_data()
        
        # remove cache if requested
        if clear_cache:
            for adapter in self.m_adapters.values():
                adapter.clear_cache()
                
    @check_mutable
    def update_adapters(self) -> None:
        ''' update adapter states, to make the changes effective
        '''
        assert self._retrieve_internal_data() is not None, f'No internal data found for {self.__class__.__name__} in unet. Please install adapters first.'
        weights = {key: val.weight for key, val in sorted(self.m_adapters.items()) if val.is_enabled}
        self._set_adapter_scale(weights)
        
    @check_mutable
    def remove_adapters(self) -> None:
        ''' remove all adapters from model
        '''
        names = list(self.m_adapters.keys())
        self._unload_adapters_from_model(names)
        
    @check_mutable
    def remove_adapter(self, name : str) -> None:
        ''' remove a specified adapter from model
        '''
        self._unload_adapters_from_model([name])
        
    def get_adapters(self) -> dict[str, DiffusionLoraAdapter]:
        ''' get all ip adapters
        '''
        return dict(self.m_adapters)
    
    def add_adapter(self, adapter : DiffusionLoraAdapter) -> None:
        ''' add an ip adapter to unet
        '''
        self.m_adapters[adapter.name] = adapter
        
    # handling internal data
    def _register_internal_data(self):
        self.m_unet._internal_data[self.__class__.__name__] = {
            self.DataKeys.adapters: self.m_adapters
        }
        
    def _retrieve_internal_data(self) -> dict:
        obj = self.m_unet._internal_data.get(self.__class__.__name__, None)
        return obj
    
    def _remove_internal_data(self):
        self.m_unet._internal_data.pop(self.__class__.__name__, None)
        
    