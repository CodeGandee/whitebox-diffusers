import numpy as np
from diffusers.loaders import TextualInversionLoaderMixin
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Callable
import torch

class DiffusionTextProcessor:
    ''' handle text processing for diffusion model
    '''
    def __init__(self) -> None:
        self.m_tokenizer : CLIPTokenizer = None
        self.m_text_encoder : Callable | CLIPTextModel = None
        self.m_text_inv = TextualInversionLoaderMixin()
        self.m_torch_device : torch.device = None
        self.m_torch_dtype : torch.dtype = None
        self.m_use_attention_mask : bool = False
        
    @classmethod
    def from_text_encoder_model(cls, tokenizer : CLIPTokenizer, model : CLIPTextModel) -> "DiffusionTextProcessor":
        ''' create text processor from text encoder model
        '''
        out = cls()
        out.m_tokenizer = tokenizer
        out.m_text_encoder = model
        out.m_torch_device = model.device
        out.m_torch_dtype = model.dtype
        
        if hasattr(model.config, 'use_attention_mask'):
            out.m_use_attention_mask = model.config.use_attention_mask
            
        return out
            
    @classmethod
    def from_text_encoder_function(cls, tokenizer : CLIPTokenizer, 
                                   encoder_func : Callable, 
                                   use_attention_mask : bool,
                                   torch_dtype : torch.dtype = None,
                                   torch_device : torch.device = None) -> "DiffusionTextProcessor":
        ''' create text processor from text encoder function, whose signature follows CLIPTextModel.__call__()
        
        parameters
        -------------
        tokenizer: CLIPTokenizer
            tokenizer to use
        encode_func: Callable
            function that takes in a string and returns a latent space vector, 
            whose signature follows CLIPTextModel.__call__()
        use_attention_mask: bool
            whether to use attention mask given by tokenizer
        torch_dtype: torch.dtype
            torch dtype to use. If None, it is up to the encode_func to do appropriate conversion
        torch_device: torch.device
            torch device to use. If None, it is up to the encode_func to do appropriate conversion
        '''
        out = cls()
        out.m_tokenizer = tokenizer
        out.m_text_encoder = encoder_func
        out.m_torch_device = torch_device
        out.m_torch_dtype = torch_dtype
        out.m_use_attention_mask = use_attention_mask
        return out
        
    def encode_text(self, text : str) -> torch.Tensor:
        ''' encode text into latent space
        '''
        # tokenize
        text_refine = self.m_text_inv.maybe_convert_prompt([text], self.m_tokenizer)[0]
        text_inputs = self.m_tokenizer(text_refine, padding='max_length', 
                                       max_length=self.m_tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt")
        
        text_ids : torch.Tensor = text_inputs.input_ids
        if self.m_use_attention_mask:
            attention_mask : torch.Tensor = text_inputs.attention_mask
        else:
            attention_mask = None
            
        if self.m_torch_dtype is not None:
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype = self.m_torch_dtype)
            
        if self.m_torch_device is not None:
            text_ids = text_ids.to(self.m_torch_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.m_torch_device)
            
        text_embeds = self.m_text_encoder(text_ids, attention_mask=attention_mask).last_hidden_state
        return text_embeds
        
        