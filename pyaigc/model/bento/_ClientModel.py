import numpy as np
import torch
import bentoml as bml
from diffusers.models.autoencoder_kl import AutoencoderKLOutput, DecoderOutput
from attrs import define, field

from pyaigc.model.DiffusionTextModelBase import IDiffusionTextModel
from pyaigc.model.DiffusionUnetBase import IDiffusionUnet
from pyaigc.model.DiffusionVaeBase import IDiffusionVae
from pyaigc.model.ControlnetBase import IControlnet, ControlNetOutput
# from igpy.common.serialization import decode_from_base64

@define(kw_only=True, init=False)
class ClientModelBase:
    ''' a thin wrapper around bentoml client
    '''
    m_model_key : str = field(default=None)
    m_client : bml.client.Client = field(default=None)
    
    def __init__(self) -> None:
        self.__attrs_init__()
        
    @property
    def model_key(self) -> str:
        return self.m_model_key
        
    @property
    def client(self) -> bml.client.Client:
        return self.m_client
        
    def connect(self, url : str):
        self.m_client = bml.client.Client.from_url(server_url=url)
        
    def close(self):
        self.m_client.close()

@define(kw_only=True, init=False)
class DiffusionControlnet(ClientModelBase, IControlnet):
    def __init__(self) -> None:
        super().__init__()
        
    @classmethod
    def from_model_key(cls, model_key:str) -> 'DiffusionControlnet':
        res : DiffusionControlnet = cls()
        res.m_model_key = model_key
        return res
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor | None,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        attention_mask: torch.Tensor | None = None,        
        cross_attention_kwargs: dict[str, any] = None
    ) -> ControlNetOutput:
        t = torch.atleast_1d(torch.tensor(timestep)).flatten()
        
        res : dict[str, np.ndarray] = self.client.run(sample=sample, timestep=t,
                                    prompt_embedding=prompt_embedding,
                                    controlnet_cond=controlnet_cond,
                                    conditioning_scale=conditioning_scale,
                                    attention_mask=attention_mask,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    return_dict=False)
        return ControlNetOutput(
            down_block_res_samples=torch.tensor(res['down_block_res_samples']),
            mid_block_res_sample=torch.tensor(res['mid_block_res_sample'])
        )

class DiffusionTextModel(ClientModelBase, IDiffusionTextModel):
    def __init__(self) -> None:
        super().__init__()
        
    @classmethod
    def from_model_key(cls, model_key:str) -> 'DiffusionTextModel':
        res : DiffusionTextModel = cls()
        res.m_model_key = model_key
        return res
    
    def encode(self, text:str) -> torch.Tensor:
        return self.client.run(text)
    
class DiffusionUnet(ClientModelBase, IDiffusionUnet):
    def __init__(self) -> None:
        super().__init__()
        
    @classmethod
    def from_model_key(cls, model_key:str) -> 'DiffusionUnet':
        res : DiffusionUnet = cls()
        res.m_model_key = model_key
        return res
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor,
        cross_attention_kwargs: dict[str, any] = None,
        down_block_additional_residuals: tuple[torch.Tensor] = None,
        mid_block_additional_residual: torch.Tensor = None
    ) -> torch.Tensor:
        res : np.ndarray = self.client.run(sample=sample, timestep=timestep,
                                    prompt_embedding=prompt_embedding,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    down_block_additional_residuals=down_block_additional_residuals,
                                    mid_block_additional_residual=mid_block_additional_residual,
                                    return_dict=False)
        return torch.tensor(res)
    
class DiffusionVAE(ClientModelBase, IDiffusionVae):
    def __init__(self) -> None:
        super().__init__()
        
    @classmethod
    def from_model_key(cls, model_key:str) -> 'DiffusionVAE':
        res : DiffusionVAE = cls()
        res.m_model_key = model_key
        return res
        
    def encode(self, x: torch.FloatTensor) -> AutoencoderKLOutput:
        _res : str = self.client.run_vae_encode(model_key = self.m_model_key, x=x)
        res : AutoencoderKLOutput = decode_from_base64(_res)
        return res
    
    def decode(self, z: torch.FloatTensor) -> DecoderOutput:
        _res : np.ndarray = self.m_client.run_vae_decode(model_key = self.m_model_key, z=z)
        return DecoderOutput(sample=torch.tensor(_res))
    