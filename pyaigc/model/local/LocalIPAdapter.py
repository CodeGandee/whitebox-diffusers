# ip-adapter and encoders
import numpy as np
import torch
from attrs import define, field

from pyaigc.model.local.LocalAdapter import (
    DiffusionAdapter,
    AdapterType,
)

from pyaigc.model.ImageEncoderBase import (
    IImageEncoder,
    ImageEncodeOutput,
)

class DiffusionIPAdapter(DiffusionAdapter):
    ''' ip adapter for diffusion model
    '''
    def __init__(self, name: str,
                 sd_version: str,
                 **kwargs) -> None:

        super().__init__(name=name, sd_version=sd_version,
                         adapter_type=AdapterType.ip_adapter,
                         **kwargs)