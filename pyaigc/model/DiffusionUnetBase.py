import numpy as np
import torch
from pyaigc.model.ModelBase import IModel

from attrs import define, field
import pyaigc.GlobalConfig as C

@define(kw_only=True)
class UnetOutput:
    sample : torch.Tensor = field()

class IDiffusionUnet(IModel):    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor,
        cross_attention_kwargs: dict[str, any] = None,
        down_block_additional_residuals: tuple[torch.Tensor] = None,
        mid_block_additional_residual: torch.Tensor = None,
        added_cond_kwargs : dict[str, torch.Tensor] = None
    ) -> UnetOutput:
        raise NotImplementedError()
    
    # additional info that must be provided by config
    @property
    def num_latent_channels(self) -> int:
        return self.config['in_channels']
    
    @property
    def sd_version(self) -> str:
        ''' stable diffusion model version, see GlobalConfig.SDVersion
        '''
        return C.SDVersion.version_from_model_key(self.model_key)
    
    def get_latent_shape_by_ouptut_size(self, width: int, height: int) -> tuple[int, int, int, int]:
        ''' get latent shape by output image width and height
        
        return
        ----------
        tuple[int, int, int, int]
            shape of latent tensor, following pytorch convention,
            in the form of (batch_size, num_latent_channels, height, width)
        '''
        
        # FIXME: divisible by 8 is required by autoencoder used by SD 1.5, may change in the future
        assert width % 8 == 0, "width must be divisible by 8"
        assert height % 8 == 0, "height must be divisible by 8"
        return (1, self.num_latent_channels, height // 8, width // 8)
    
    