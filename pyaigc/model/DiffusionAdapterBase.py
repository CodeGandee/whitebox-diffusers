# adapters used with diffusion process, mainly the unet model
import numpy as np
import torch
from typing import Any
from pyaigc.model.ModelBase import IModel

import pyaigc.GlobalConfig as C

class IDiffusionAdapter(IModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def set_name(self, name : str) -> None:
        ''' set the name of this adapter
        '''
        raise NotImplementedError()
    
    @property
    def name(self) -> str:
        ''' name of this adapter, unique in a diffusion model
        '''
        raise NotImplementedError()
    
    @property
    def adapter_type(self) -> str:
        ''' type of this adapter, see AdapterType
        '''
        raise NotImplementedError()
    
    @property
    def is_enabled(self) -> bool:
        ''' whether this adapter is enabled
        '''
        raise NotImplementedError()
    
    @property
    def weight(self) -> float:
        ''' weight of this adapter in a pipeline
        '''
        raise NotImplementedError()
    
    @property
    def get_data(self) -> dict[str, Any]:
        ''' get all tensors of this adapter, the actual content depends on the adapter type
        '''
        raise NotImplementedError() 
    
    @property
    def metadata(self) -> dict[str, str] | None:
        ''' metadata of this adapter
        '''
        return None
    
    @property
    def config(self) -> dict[str, str] | None:
        ''' config of this adapter
        '''
        return self.metadata
    
class IAdapterSupportMixin:
    ''' for models that support adapters
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def add_adapter(self, adapter: IDiffusionAdapter, replace_if_exists : bool = False) -> None:
        ''' add an adapter to the model.
        
        parameters
        ----------
        adapter : IDiffusionAdapter
            adapter to add
        replace_if_exists : bool
            whether to replace the adapter if it already exists. If False, raise an error if the adapter already exists.
        '''
        raise NotImplementedError()
    
    def remove_adapter(self, name: str) -> None:
        ''' remove an adapter from the model
        '''
        raise NotImplementedError()
    
    def remove_all_adapters(self) -> None:
        ''' remove all adapters from the model
        '''
        raise NotImplementedError()
    
    def get_adapter_by_name(self, name: str) -> IDiffusionAdapter:
        ''' get an adapter by name, return None if not found
        '''
        raise NotImplementedError()
    
    def get_adapters(self) -> dict[str, IDiffusionAdapter]:
        ''' get all adapters
        '''
        raise NotImplementedError()
    
    def apply_adapters(self) -> None:
        ''' apply all adapters to the model, updating the model state
        '''
        raise NotImplementedError()
    