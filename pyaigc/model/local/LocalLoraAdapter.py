import numpy as np
import torch
from attrs import define, field
from typing import Any
from typing_extensions import Self
from pyaigc.model.local.LocalAdapter import DiffusionAdapter

import pyaigc.GlobalConfig as C

from pyaigc.GlobalConfig import (
    AdapterType, SafetensorData
)



class DiffusionLoraAdapter(DiffusionAdapter):
    ''' a lora adapter for diffusion model
    '''

    def __init__(self, name: str,
                 sd_version: str,
                 model_key: str = None,
                 is_enabled: bool = True,
                 state_dict: dict[str, torch.Tensor] = None,
                 metadata: dict[str, str] = None,
                 weight: float = 1.0) -> None:

        super().__init__(name=name, sd_version=sd_version,
                         model_key=model_key,
                         adapter_type=C.AdapterType.lora,
                         is_enabled=is_enabled,
                         state_dict=state_dict,
                         metadata=metadata,
                         weight=weight)
        

    @classmethod
    def from_safetensor_data(cls, name: str,
                             sd_version: str,
                             tensors: dict[str, torch.Tensor],
                             metadata: dict[str, str] | None,
                             weight: float = 1.0) -> Self:
        ''' create a lora adapter from state dict.

        parameters
        ------------
        name : str
            name of this adapter
        tensors : dict[str, torch.Tensor]
            state dict of this adapter, possibly loaded from a safetensor file
        metadata : dict[str, str]
            metadata of this adapter, if any, you can obtain this info from safetensor file
        weight : float
            weight of this adapter in a pipeline
        sd_version : str
            stable diffusion model type, see GlobalConfig.SDVersion

        return
        --------
        out : DiffusionLoraAdapter
            a lora adapter
        '''
        return cls(name=name,
                   sd_version=sd_version,
                   state_dict=tensors,
                   metadata=metadata,
                   weight=weight)