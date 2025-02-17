import numpy as np
import torch
from attrs import define, field
from typing import Any
from typing_extensions import Self

import igpy.ml.util_diffusers as ut_diffusers

from pyaigc.model.DiffusionAdapterBase import IDiffusionAdapter
from pyaigc.model.ModelBase import MutabilityMixin

import pyaigc.GlobalConfig as C

from pyaigc.GlobalConfig import (
    AdapterType, SafetensorData
)


class DiffusionAdapter(MutabilityMixin, IDiffusionAdapter):
    def __init__(self, name: str,
                 sd_version: str,
                 model_key: str = None,
                 adapter_type: str = None,
                 is_enabled: bool = True,
                 state_dict: dict[str, Any] = None,
                 metadata: dict[str, str] = None,
                 weight: float = 1.0) -> None:
        ''' create a diffusion adapter

        parameters
        ------------
        name : str
            name of this adapter
        state_dict : dict[str, Any] | None
            state dict of this adapter, possibly loaded from a safetensor file
        metadata : dict[str, str] | None
            metadata of this adapter, if any, you can obtain this info from safetensor file
        weight : float
            weight of this adapter in a pipeline
        sd_version : str
            stable diffusion model type, see GlobalConfig.SDVersion
        adapter_type : str
            type of this adapter, see GlobalConfig.AdapterType
        model_key : str
            model key of this adapter
        is_enabled : bool
            whether this adapter is enabled now
        '''
        super().__init__()

        self.m_name: str = name
        self.m_state_dict: dict[str, Any] = state_dict
        self.m_metadata: dict[str, str] = metadata
        self.m_weight: float = weight
        self.m_sd_version: str = sd_version
        self.m_adapter_type: str = adapter_type
        self.m_model_key: str = model_key
        self.m_is_enabled: bool = is_enabled

    @property
    def metadata(self) -> dict[str, str] | None:
        return self.m_metadata

    @property
    def name(self) -> str:
        return self.m_name

    @property
    def adapter_type(self) -> str:
        return self.m_adapter_type

    @property
    def is_enabled(self) -> bool:
        return self.m_is_enabled

    @property
    def weight(self) -> float:
        return self.m_weight

    @property
    def model_key(self) -> str:
        return self.m_model_key

    @property
    def sd_version(self) -> str:
        return self.m_sd_version
    
    @property
    def device(self) -> torch.device:
        t = self._get_a_state_tensor(self.m_state_dict)
        if t is not None:
            return t.device
        
    @property
    def dtype(self) -> torch.dtype:
        t = self._get_a_state_tensor(self.m_state_dict)
        if t is not None:
            return t.dtype
    
    def _get_a_state_tensor(self, tensor_dict : dict) -> torch.Tensor:
        ''' get a tensor from state dict
        '''
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                return val
            elif isinstance(val, dict):
                return self._get_a_state_tensor(val)

    def get_data(self) -> dict[str, Any]:
        return self.m_state_dict

    def set_weight(self, weight: float) -> None:
        self.m_weight = weight

    def set_enabled(self, enabled: bool) -> None:
        self.m_is_enabled = enabled
        
    def set_name(self, name: str) -> None:
        self.m_name = name
        
    def shallow_copy(self) -> Self:
        ''' make a shallow copy of this adapter, so that you can modify the name, weight, enabled status, etc.
        Note that the tensors are not copied, so you should not modify the tensors in the copy.
        '''
        import copy
        obj = copy.copy(self)
        obj.m_metadata = copy.copy(self.m_metadata)
        obj.m_state_dict = copy.copy(self.m_state_dict)
        return obj
    
    def clear_cache(self):
        ''' clear cached state dict. This is useful when you want to save memory,
        the state dict is not needed after adapter installation.
        '''
        self.m_state_dict = {}

