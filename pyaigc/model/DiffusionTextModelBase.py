import numpy as np
import torch
from attrs import define, field
import attrs.validators as av
from igpy.common.validators import (
    attrs_check_all_3d_array, 
    attrs_check_3d_array,
    attrs_check_2d_array
)

from transformers.models.clip.modeling_clip import CLIPTextModelOutput, CLIPTextModel
from .ModelBase import IModel

@define(kw_only=True)
class TextModelOutput:
    # (batch, n_token, feature_dim), the final per-word embedding
    word_embedding : torch.Tensor | None = field(default=None, validator=av.optional(attrs_check_3d_array))
    
    # (batch, feature_dim), the final 1d embedding of the text
    projected_embedding : torch.Tensor | None = field(default=None, validator=av.optional(attrs_check_2d_array))
    
    # (batch, n_token, feature_dim), hidden states of the text model, 
    # can be considered as per-word embedding with different granularity
    hidden_states : list[torch.Tensor] | None = field(default=None, validator=av.optional(attrs_check_all_3d_array))
    
class IDiffusionTextModel(IModel):
    ''' text embedding for diffusion model
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def encode(self, text:str,
               embedding_layer_index : int = -1,
               output_hidden_states = False,
               tokenizer_params : dict[str, object] = None
               ) -> TextModelOutput:
        ''' encode text into embedding
        
        parameters
        -----------
        text : str
            text to encode
        embedding_layer_index : int
            use which hidden state layer as word embedding, -1 is the last layer, 
            -2 is the second last layer, and so on.
        with_hidden_states : bool
            whether to return all hidden states
        tokenizer_params : dict[str, object]
            parameters to pass to the tokenizer
            
        return
        -------
        output : TextModelOutput
            output of the text model. Note that output.hidden_states[k] is the hidden state of hidden_state_layers[k],
            if hidden_state_layers is not None.
        '''
        raise NotImplementedError()
    
    def tokenize(self, text:str) -> torch.Tensor:
        raise NotImplementedError()