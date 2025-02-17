import numpy as np
import torch
from attrs import define, field, asdict

from pyaigc.model.DiffusionVaeBase import IDiffusionVae
from pyaigc.model.DiffusionTextModelBase import IDiffusionTextModel
from pyaigc.model.DiffusionUnetBase import IDiffusionUnet
from pyaigc.model.DiffusionScheduler import DiffusionScheduler
from pyaigc.model.ControlnetBase import IControlnet
from pyaigc.model.ImageEncoderBase import IImageEncoder
from pyaigc.model.DiffusionAdapterBase import IDiffusionAdapter

@define(kw_only=True, eq=False)
class DiffusionModelBundle:
    ''' models that constitute the stable diffusion process '''    
    vae : IDiffusionVae | None = field(default=None)
    unet : IDiffusionUnet | None = field(default=None)
    text_model : IDiffusionTextModel | None = field(default=None)
    text_model_2 : IDiffusionTextModel | None = field(default=None)
    scheduler : DiffusionScheduler | None = field(default=None)
    controlnets : dict[str, IControlnet] = field(factory=dict)
    ip_image_encoder : IImageEncoder | None = field(default=None)
    ip_adapters : dict[str, IDiffusionAdapter] = field(factory=dict)