import torch
import numpy as np
import einops
from attrs import define, field
import attrs.validators as av
import dataclasses as dc
from typing import Literal
from typing_extensions import Self

from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import (
    CLIPTextModel, CLIPTokenizer, 
    CLIPTextModelWithProjection,
)
from transformers.models.clip.modeling_clip import (
    CLIPTextModelOutput, 
    BaseModelOutputWithPooling
)

from diffusers.loaders import TextualInversionLoaderMixin

from pyaigc.model.DiffusionUnetBase import IDiffusionUnet, UnetOutput
from pyaigc.model.DiffusionVaeBase import IDiffusionVae, VaeEncodeOutput
from pyaigc.model.DiffusionTextModelBase import IDiffusionTextModel, TextModelOutput
from pyaigc.model.ControlnetBase import IControlnet, ControlnetModelOutput
from pyaigc.model.ModelBase import IModel, MutabilityMixin
from pyaigc.model.local.LocalAdapter import (
    IDiffusionAdapter,
    AdapterType
)
from pyaigc.model.local.LocalLoraAdapter import DiffusionLoraAdapter

import pyaigc.GlobalConfig as C

from igpy.common.shortfunc import to_4d_tensor
# import pyaigc.TextHandling as th

import igpy.common.shortfunc as sf

class TorchModuleMixin(torch.nn.Module):
    ''' mixin for torch module, implementing part of IModel interface
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_model_key : str = None
        
    @property
    def model_key(self) -> str:
        return self.m_model_key
    
class LocalModelBase(MutabilityMixin, TorchModuleMixin):
    ''' base of all local models
    '''
    def __init__(self, *args, **kwargs) -> None:
        # data for internal use, it may not be cloned when the model is cloned
        self.m_internal_data : dict[str, object] = {}
        
        super().__init__(*args, **kwargs)    
        
    @property
    def _internal_data(self) -> dict[str, object]:
        ''' internal data, it may not be cloned when the model is cloned
        '''
        return self.m_internal_data
    
class DiffusionUnet(LocalModelBase, IDiffusionUnet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.m_model : UNet2DConditionModel = None
        
    @classmethod
    def from_diffuser(cls, unet : UNet2DConditionModel, 
                      model_key : str = None) -> 'DiffusionUnet':
        out = cls()
        out.m_model = unet
        out.m_model_key = model_key

        return out
    
    @property
    def config(self) -> dict[str, object]:
        if self.m_model is not None:
            return self.m_model.config.__dict__
        
    @property
    def device(self) -> str | None:
        if self.m_model is not None:
            return str(self.m_model.device)
        
    def get_internal_model(self) -> UNet2DConditionModel:
        return self.m_model
    
    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor,
        cross_attention_kwargs: dict[str, any] = None,
        down_block_additional_residuals: tuple[torch.Tensor] = None,
        mid_block_additional_residual: torch.Tensor = None,
        added_cond_kwargs : dict[str, torch.Tensor] = None
    ) -> UnetOutput:
        
        target_dtype = self.m_model.dtype
        target_device = self.m_model.device
        
        sample_ = sample.to(dtype=target_dtype, device=target_device)
        prompt_embedding_ = prompt_embedding.to(dtype=target_dtype, device=target_device)
        
        down_block_additional_residuals_ = down_block_additional_residuals
        if down_block_additional_residuals is not None:
            down_block_additional_residuals_ = tuple([x.to(dtype=target_dtype, device=target_device) for x in down_block_additional_residuals])
            
        mid_block_additional_residual_ = mid_block_additional_residual
        if mid_block_additional_residual is not None:
            mid_block_additional_residual_ = mid_block_additional_residual.to(dtype=target_dtype, device=target_device)
            
        # recusively convert all tensors to target dtype and device
        def convert_all_tensor_recurse(dict_of_tensor: dict):
            for k,v in dict_of_tensor.items():
                if isinstance(v, dict):
                    convert_all_tensor_recurse(v)
                elif isinstance(v, torch.Tensor):
                    dict_of_tensor[k] = v.to(dtype=target_dtype, device=target_device)
            
        import copy
        if added_cond_kwargs is not None:
            added_cond_kwargs_ = copy.copy(added_cond_kwargs)
            convert_all_tensor_recurse(added_cond_kwargs_)
        else:
            added_cond_kwargs_ = added_cond_kwargs
            # added_cond_kwargs_ = {k: v.to(dtype=target_dtype, device=target_device) for k,v in added_cond_kwargs.items()}
            
        # timestep_ = torch.tensor(np.atleast_1d(timestep), device=target_device)
        if not isinstance(timestep, torch.Tensor):
            timestep_ = torch.tensor(np.atleast_1d(timestep), device=target_device)
        else:
            timestep_ = torch.atleast_1d(timestep).to(device=target_device)
        
        res = self.m_model(
            sample=sample_,
            timestep=timestep_,
            encoder_hidden_states=prompt_embedding_,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals_,
            mid_block_additional_residual=mid_block_additional_residual_,
            added_cond_kwargs=added_cond_kwargs_,
            return_dict=False
        )
        
        return UnetOutput(sample=res[0])
    
class DiffusionVae(LocalModelBase, IDiffusionVae):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_model : AutoencoderKL = None
        self.m_image_proc : VaeImageProcessor = None
        self.m_scale_factor : float = 1.0
    
    @property
    def config(self) -> dict[str, object]:
        if self.m_model is not None:
            return self.m_model.config.__dict__
        
    @property
    def device(self) -> str | None:
        if self.m_model is not None:
            return str(self.m_model.device)
    
    @classmethod
    def from_model(cls, model : AutoencoderKL, model_key : str = None) -> 'DiffusionVae':
        out = cls()
        out.m_model = model
        out.m_model_key = model_key
        out.m_image_proc = VaeImageProcessor(do_convert_rgb=True, do_resize=False)
        
        return out
    
    @property
    def latent_scaling_factor(self) -> float:
        ''' when training stable diffusion, the vae latent is scaled. Specifically, 
        latent_of_diffusion = latent_of_vae * latent_scaling_factor.
        So, when using the vae to encode image, the latent should be scaled by this factor,
        when using the vae to decode latent, the latent should be divided by this factor.
        
        Note that this only applies to latent generated by stable diffusion. For those generated by vae encode/decode,
        you do not need to scale it.
        '''
        return self.m_model.config.get('scaling_factor', 1.0)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, input_data_layout : str = 'nchw') -> VaeEncodeOutput:
        
        # expand to 4d tensor by appending/prepending dimensions
        x = to_4d_tensor(x, input_layout=input_data_layout, output_layout='nchw')
        
        # if uint8, convert to float
        if x.dtype == torch.uint8:
            x = x.to(dtype=self.m_model.dtype, device=self.m_model.device) / 255.0
        else:
            x = x.to(dtype=self.m_model.dtype, device=self.m_model.device)
            
        x = VaeImageProcessor.normalize(x)
        x = x.to(dtype=self.m_model.dtype, device=self.m_model.device)
        latent_dist : DiagonalGaussianDistribution = self.m_model.encode(x, return_dict=False)[0]
        
        out = VaeEncodeOutput(mean=latent_dist.mean, std=latent_dist.std)
        return out
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor, 
               output_dtype : torch.dtype | np.dtype = None, 
               output_data_layout : str = 'nchw') -> torch.Tensor:

        # output in nchw format
        z = z.to(device=self.m_model.device, dtype=self.m_model.dtype)
        out : torch.Tensor = self.m_model.decode(z, return_dict=False)[0]
        
        # denormalize
        out = VaeImageProcessor.denormalize(out)
        
        # convert to torch.dtype
        if isinstance(output_dtype, np.dtype):
            output_dtype = torch.from_numpy(np.zeros(0, dtype=output_dtype)).dtype
        
        # convert to output layout
        if output_data_layout == 'nhwc':
            out = einops.rearrange(out, 'n c h w -> n h w c')
            
        # convert to output dtype
        if output_dtype == torch.uint8 or output_dtype == np.uint8:
            out = (out.clip(0,1.0) * 255).to(dtype=torch.uint8)
        else:
            out = out.to(dtype=output_dtype)
            
        return out

class DiffusionTextModel(LocalModelBase, IDiffusionTextModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_model : CLIPTextModel | CLIPTextModelWithProjection = None
        self.m_text_tokenizer : DiffusionTextTokenizer = None
    
    @classmethod
    def from_clip(cls, tokenizer : CLIPTokenizer,
                  encoder_model : CLIPTextModel,
                  model_key:str = None) -> 'DiffusionTextModel':
        out = DiffusionTextModel()
        out.m_model = encoder_model
        out.m_model_key = model_key
        out.m_text_tokenizer = DiffusionTextTokenizer(
            tokenizer=tokenizer, device=encoder_model.device, 
            dtype=encoder_model.dtype)
        return out
    
    @property
    def config(self) -> dict[str, object]:
        ''' config of encoder and tokenizer
        '''
        return {
            'encoder': self.m_model.config.__dict__,
            'tokenizer':self.m_text_tokenizer.m_tokenizer.__dict__
        }
        
    @property
    def device(self) -> str | None:
        if self.m_model is not None:
            return str(self.m_model.device)
        
    def get_tokenizer(self) -> 'DiffusionTextTokenizer':
        return self.m_text_tokenizer
    
    @torch.no_grad()
    def encode(self, text:str,
               embedding_layer_index : int = -1,
               output_hidden_states = False,
               tokenizer_params : dict[str, object] = None
               ) -> TextModelOutput:
        ''' encode text to tensor
        '''
        assert embedding_layer_index < 0, 'layer index should be negative'
        
        if tokenizer_params is None:
            tokenizer_params = {}
            
        tok_result = self.m_text_tokenizer.tokenize(text, **tokenizer_params)
        token_ids = tok_result.token_ids.to(device=self.m_model.device)
        attention_mask = tok_result.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.m_model.device, dtype=self.m_model.dtype)    
        
        if isinstance(self.m_model, CLIPTextModelWithProjection):
            res : CLIPTextModelOutput = self.m_model(token_ids,
                                                     output_hidden_states = True, 
                                                     attention_mask=attention_mask)
            if embedding_layer_index == -1:
                word_embed : torch.Tensor = res.last_hidden_state
            else:
                word_embed : torch.Tensor = res.hidden_states[embedding_layer_index]
                
            out = TextModelOutput(
                word_embedding=word_embed,
                projected_embedding=res.text_embeds,
                hidden_states=res.hidden_states if output_hidden_states else None
            )
            
        elif isinstance(self.m_model, CLIPTextModel):
            res_ : BaseModelOutputWithPooling = self.m_model(token_ids,
                                                             output_hidden_states = True, 
                                                             attention_mask=attention_mask)
            
            if embedding_layer_index == -1:
                word_embed : torch.Tensor = res_.last_hidden_state
            else:
                word_embed : torch.Tensor = res_.hidden_states[embedding_layer_index]
                
            out = TextModelOutput(
                word_embedding=word_embed,
                projected_embedding=res_.pooler_output,
                hidden_states=res_.hidden_states if output_hidden_states else None
            )
                
        else:
            raise ValueError(f'unknown model type {type(self.m_model)}')
            
        # out : torch.Tensor = self.m_model(token_ids, attention_mask=attention_mask).last_hidden_state
        return out
    
class DiffusionControlnet(LocalModelBase, IControlnet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.m_model : ControlNetModel = None
        self.m_image_proc : VaeImageProcessor = None
    
    @property
    def device(self) -> str | None:
        if self.m_model is not None:
            return str(self.m_model.device)
        
    @property
    def config(self) -> dict[str, object]:
        return self.m_model.config.__dict__
    
    @classmethod
    def from_model(cls, model : ControlNetModel, model_key : str = None) -> 'DiffusionControlnet':
        out = cls()
        out.m_model = model
        out.m_model_key = model_key
        out.m_image_proc = VaeImageProcessor(do_resize=False, do_convert_rgb=True, do_normalize=False)
        return out
    
    @torch.no_grad()
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
        prompt_embedding: torch.Tensor,
        condition: torch.Tensor,
        condition_data_layout : Literal['nchw','nhwc'],
        conditioning_scale: float = 1.0,
        attention_mask: torch.Tensor | None = None,        
        cross_attention_kwargs: dict[str, object] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> ControlnetModelOutput:
        
        target_dtype = self.m_model.dtype
        target_device = self.m_model.device
        
        sample_ = sample.to(dtype=target_dtype, device=target_device)
        prompt_embedding_ = prompt_embedding.to(dtype=target_dtype, device=target_device)
        
        # convert to nchw format
        condition = to_4d_tensor(condition, input_layout=condition_data_layout, output_layout='nchw')
        if condition.dtype == torch.uint8:
            controlnet_cond_ = condition.to(dtype=target_dtype, device=target_device) / 255.0
        else:
            controlnet_cond_ = condition.to(dtype=target_dtype, device=target_device)
            
        # convert to rgb
        if controlnet_cond_.shape[1] == 1:
            controlnet_cond_ = controlnet_cond_.repeat(1,3,1,1)
            
        attention_mask_ = attention_mask
        if attention_mask is not None:
            attention_mask_ = attention_mask.to(dtype=target_dtype, device=target_device)
            
        added_cond_kwargs_ = added_cond_kwargs
        if added_cond_kwargs is not None:
            added_cond_kwargs_ = {k: v.to(dtype=target_dtype, device=target_device) for k,v in added_cond_kwargs.items()}
        
        timestep_ = torch.tensor(np.atleast_1d(timestep), device=target_device)
        conditioning_scale_ = torch.tensor(np.atleast_1d(conditioning_scale), device=target_device, dtype=target_dtype)
        conditioning_scale_ = conditioning_scale_.view(-1, 1, 1, 1)
        
        res = self.m_model(
                sample=sample_,
                timestep=timestep_,
                encoder_hidden_states=prompt_embedding_,
                controlnet_cond=controlnet_cond_,
                conditioning_scale=conditioning_scale_,
                attention_mask=attention_mask_,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs = added_cond_kwargs_,
                return_dict=False)
        
        return ControlnetModelOutput(
            down_block_res_samples=res[0],
            mid_block_res_sample=res[1])
    
class DiffusionTextTokenizer(TorchModuleMixin):
    ''' tokenize the text for diffusion model    
    '''
    def __init__(self, tokenizer : CLIPTokenizer = None,
                 use_attention_mask : bool = False,
                 device : torch.device = C.DEFAULT_TORCH_DEVICE,
                 dtype : torch.dtype = C.DEFAULT_TORCH_DTYPE) -> None:
        super().__init__()
        
        self.m_tokenizer : CLIPTokenizer = tokenizer
        self.m_use_attention_mask : bool = use_attention_mask
        self.m_torch_device : torch.device = device
        self.m_torch_dtype : torch.dtype = dtype
        self.m_text_inv : TextualInversionLoaderMixin = TextualInversionLoaderMixin()
    
    @define(kw_only=True, slots=False, unsafe_hash=True)
    class TokenizedOutput:
        ''' output of tokenization
        '''
        token_ids : torch.Tensor | None
        attention_mask : torch.Tensor | None = None
        
        def to(self, device : torch.device):
            if self.token_ids is not None:
                self.token_ids = self.token_ids.to(device=device)
            if self.attention_mask is not None:
                self.attention_mask = self.attention_mask.to(device=device)
    
    def tokenize(self, text:str) -> 'DiffusionTextTokenizer.TokenizedOutput':
        ''' tokenize text
        '''
        text_refine = self.m_text_inv.maybe_convert_prompt([text], self.m_tokenizer)[0]
        text_inputs = self.m_tokenizer(text_refine, padding='max_length', 
                                       max_length=self.m_tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt")
        
        text_ids : torch.Tensor = text_inputs.input_ids.to(device=self.m_torch_device)
        
        if self.m_use_attention_mask:
            attention_mask : torch.Tensor = text_inputs.attention_mask
        else:
            attention_mask = None
            
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.m_torch_device, dtype=self.m_torch_dtype)
            
        return self.TokenizedOutput(token_ids=text_ids, attention_mask=attention_mask)
        