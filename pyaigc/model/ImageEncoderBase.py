import torch
from pyaigc.model.ModelBase import IModel
from attrs import define, field

@define(kw_only=True, eq=False, slots=False)
class ImageEncodeOutput:
    ''' the output of clip image encoder, see CLIPVisionModelOutput in transformers package
    '''
    # The image embeddings obtained by applying the projection layer to the pooler_output
    # (batch_size, output_dim)
    image_embedding : torch.Tensor | None = field(default=None)
    
    # Sequence of hidden-states at the output of the last layer of the model
    # (batch_size, sequence_length, hidden_size)
    last_hidden_state : torch.Tensor | None = field(default=None)
    
    # Hidden-states of the model at the output of each layer plus the optional initial embedding outputs
    # hidden states at the later layer come first in the list
    hidden_states : list[torch.Tensor] | None = field(default=None)
    
    # (batch_size, num_heads, sequence_length, sequence_length)
    # Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads
    attentions : list[torch.Tensor] | None = field(default=None)
    
    # left for extensions
    more : dict | None = field(default=None)

class IImageEncoder(IModel):
    ''' encode image into feature
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def encode(self, x: torch.Tensor, 
               input_data_layout : str = 'nchw',
               with_hidden_states : bool = False,
               with_attentions : bool = False,
               **kwargs) -> ImageEncodeOutput:
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
        
        raise NotImplementedError()
        
    