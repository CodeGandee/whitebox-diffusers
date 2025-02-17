# image encoders
import torch
import numpy as np
from attrs import define, field

from transformers import (
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor,
)
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput

from pyaigc.model.local.LocalModel import LocalModelBase
from pyaigc.model.ImageEncoderBase import (
    IImageEncoder,
    ImageEncodeOutput,
)

class ImageEncoderByCLIP(LocalModelBase, IImageEncoder):
    def __init__(self) -> None:
        super().__init__()
        
        # the clip model
        self.m_model : CLIPVisionModelWithProjection = None
        
        # image must be preprocessed before passing to the model
        self.m_preprocessor : CLIPImageProcessor = CLIPImageProcessor()
    
    @property
    def model(self) -> CLIPVisionModelWithProjection:
        return self.m_model
    
    @property
    def config(self) -> dict:
        return self.m_model.config.to_dict()
    
    @property
    def device(self) -> torch.device:
        return self.m_model.device
    
    @classmethod
    def from_encoder_model(cls, model : CLIPVisionModelWithProjection) -> 'ImageEncoderByCLIP':
        ''' create an instance from a clip model
        '''
        encoder = cls()
        encoder.m_model = model
        return encoder
        
    def preprocess_image(self, x: torch.Tensor, input_data_layout : str = 'nchw') -> torch.Tensor:
        ''' preprocess image tensor to clip input format.
        
        parameters
        ---------------
        x : torch.Tensor
            the image tensor, can be uint8 or float16/32. For float image, the pixel value range is within [0,1].
        input_data_layout : str
            the layout of input data, can be 'nchw' or 'nhwc'
            
        return
        ----------
        out : torch.Tensor
            the preprocessed image tensor
        '''
        assert input_data_layout in ('nchw', 'nhwc'), f'unsupported input_data_layout: {input_data_layout}'
        
        if input_data_layout == 'nchw':
            n_channel = x.shape[1]
        else:
            n_channel = x.shape[-1]
            
        assert n_channel in (1, 3), f'unsupported channel number: {n_channel}'
            
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            # assuming the pixel value is within [0,1]
            z = self.m_preprocessor(x, return_tensors='pt', do_rescale=False).pixel_values
        elif x.dtype == torch.uint8:
            # it can handle hwc, nchw, nhwc format automatically
            z = self.m_preprocessor(x, return_tensors='pt', do_rescale=True).pixel_values
        else:
            raise ValueError(f'unsupported image data type: {x.dtype}')
        
        return z

    @torch.no_grad()
    def encode(self, x: torch.Tensor, 
               input_data_layout : str = 'nchw',
               with_hidden_states : bool = False,
               with_attentions : bool = False) -> ImageEncodeOutput:
        ''' encode image to latent space.
        
        parameters
        ---------------
        x : torch.Tensor
            image tensor in batch format, can be uint8 or float16/32. For float image, 
            the pixel value range is within [0,1].
        input_data_layout : str
            the layout of input data, can be 'nchw' or 'nhwc'
        with_hidden_states : bool
            whether to return hidden states
        with_attentions : bool
            whether to return attentions
            
        return
        ----------
        ImageEncodeOutput
            the encode output, in nchw layout
        '''
        # preprocess image
        img = self.preprocess_image(x, input_data_layout = input_data_layout)
        
        dtype = self.m_model.parameters().__next__().dtype
        device = self.m_model.parameters().__next__().device
        
        # encode, return_dict=False because it will have problem when deploying to bentoml
        _img = img.to(dtype=dtype, device=device)
        tmp = self.m_model(_img, 
                           output_hidden_states=with_hidden_states, 
                           output_attentions=with_attentions,
                           return_dict=False)
        _out : CLIPVisionModelOutput = CLIPVisionModelOutput(*tmp)
        
        out = ImageEncodeOutput(
            image_embedding=_out.image_embeds,
            last_hidden_state=_out.last_hidden_state,
            hidden_states=list(_out.hidden_states) if _out.hidden_states is not None else None,
            attentions=list(_out.attentions) if _out.attentions is not None else None,
        )
        
        return out