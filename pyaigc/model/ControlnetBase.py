import numpy as np
import torch
from typing import Literal
from attrs import define, field
from .ModelBase import IModel
# from diffusers.models import ControlNetModel
# from diffusers.models.controlnet import ControlNetOutput

@define(kw_only=True)
class ControlnetModelOutput:
    down_block_res_samples: list[torch.Tensor] = field()
    mid_block_res_sample: torch.Tensor = field()

class IControlnet(IModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor,
        condition: torch.Tensor,
        condition_data_layout : Literal['nchw','nhwc'],
        conditioning_scale: float = 1.0,
        attention_mask: torch.Tensor | None = None,        
        cross_attention_kwargs: dict[str, object] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> ControlnetModelOutput:
        raise NotImplementedError()
    
    @property
    def model_key(self)->str:
        ''' a unique name of this type of model
        '''
        raise NotImplementedError()
    
    # def image_to_model_input(self, image : np.ndarray) -> torch.FloatTensor:
    #     ''' convert visible image to model input tensor
    #     '''
    #     raise NotImplementedError()