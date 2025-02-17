import numpy as np
import torch
from attrs import define, field
import attrs.validators as av

import diffusers.models.embeddings as df_emb
from diffusers.models.attention_processor import (
    IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
)

from pyaigc.GlobalConfig import ImageTensor
from pyaigc.model.local.LocalImageEncoder import (
    ImageEncoderByCLIP, ImageEncodeOutput
)

from pyaigc.model.local.LocalIPAdapter import(
    DiffusionIPAdapter
)

from pyaigc.model.local.LocalModel import(
    DiffusionUnet
)

ProjectionLayerType = df_emb.ImageProjection \
            | df_emb.IPAdapterFullImageProjection \
            | df_emb.IPAdapterPlusImageProjection \
            | df_emb.MultiIPAdapterImageProjection

@define(kw_only=True, eq=False)
class ImagePromptEmbedding:
    positive : torch.Tensor = field(validator=av.instance_of(torch.Tensor))
    negative : torch.Tensor = field(validator=av.instance_of(torch.Tensor))

class IPAdapterHelper:
    ''' helper class for ip adapter, which installs and manages ip adapters in unet
    '''
    def __init__(self) -> None:
        self.m_unet : DiffusionUnet = None
        self.m_ip_adapters : dict[str, DiffusionIPAdapter] = {}
        self.m_image_encoder : ImageEncoderByCLIP = None
        
    @classmethod
    def from_unet(cls, unet : DiffusionUnet, 
                  image_encoder : ImageEncoderByCLIP = None, 
                  ip_adapters : dict[str, DiffusionIPAdapter] = None) -> 'IPAdapterHelper':
        ''' create an instance from unet
        '''
        helper = cls()
        helper.m_unet = unet
        import copy
        
        if ip_adapters is not None:
            assert set(ip_adapters.keys()) == set([x.name for x in ip_adapters.values()]), 'key and name mismatch'
            helper.m_ip_adapters = copy.copy(ip_adapters)
        else:
            # read from unet
            helper.m_ip_adapters = unet._internal_data.get(cls.__name__, {})
        
        helper.m_image_encoder = image_encoder
        return helper
    
    def get_adapters(self) -> dict[str, DiffusionIPAdapter]:
        ''' get all ip adapters
        '''
        return dict(self.m_ip_adapters)
    
    def add_adapter(self, adapter : DiffusionIPAdapter) -> None:
        ''' add an ip adapter to unet
        '''
        self.m_ip_adapters[adapter.name] = adapter
        
    def _add_adapters_to_model(self) -> None:
        ''' add ip adapters to unet
        '''
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dicts = []
        scale_per_adapter = []
        
        # use sorted to ensure consistent order
        for ip_name, ip_obj in sorted(self.m_ip_adapters.items()):
            ip_data = ip_obj.get_data()
            keys = list(ip_data.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")
            
            state_dicts.append(ip_data)
            scale_per_adapter.append(ip_obj.weight)
        self.m_unet.m_model._load_ip_adapter_weights(state_dicts)
        self._set_adapter_scale(scale_per_adapter)
        
    def _unload_adapters_from_model(self) -> None:
        ''' unload ip adapters from unet
        '''
        
        # check if ip adapters are loaded, if yes, unload them
        unet_internal = self.m_unet.m_model
        if self.is_adapter_loaded():
            self.m_unet.m_model.encoder_hid_proj = None
            self.m_unet.m_model.config.encoder_hid_dim_type = None
            self.m_unet.m_model.set_default_attn_processor()
                
    def is_adapter_loaded(self) -> bool:
        ''' check if ip adapter is loaded
        '''
        if self.m_unet is None:
            return False
        
        # check internal data
        has_internal_data = self._retrieve_internal_data() is not None
        unet_config = self.m_unet.m_model.config
        
        has_unet_data = ('encoder_hid_dim_type' in unet_config) and \
            unet_config.encoder_hid_dim_type == 'ip_image_proj'
        
        # they must be consistent
        assert has_internal_data == has_unet_data, f'Inconsistent internal data found in unet, the loaded ip adapters may have been changed.'
        
        return has_internal_data
        
    def _retrieve_internal_data(self) -> dict[str, DiffusionIPAdapter]:
        ''' get internal data from unet
        '''
        return self.m_unet._internal_data.get(self.__class__.__name__, None)
    
    def _register_internal_data(self) -> None:
        ''' write internal data to unet
        '''
        self.m_unet._internal_data[self.__class__.__name__] = self.m_ip_adapters
        
    def _remove_internal_data(self) -> None:
        ''' remove internal data from unet
        '''
        self.m_unet._internal_data.pop(self.__class__.__name__, None)

    def install_adapters(self, clear_cache : bool = False,
                         replace_existing : bool = False) -> None:
        ''' add ip adapters to unet, activate them
        
        parameters
        ------------
        clear_cache : bool
            whether to clear cache after installation. If True, you can only install once because
            the data cached in adapter objects will be removed once installed.
        replace_existing : bool
            whether to replace existing adapters if they are already installed. 
            Default is False, and will raise an error if there are existing adapters.
        '''
        # cannot install if there are adapters already installed
        if self.is_adapter_loaded():
            if not replace_existing:
                raise ValueError(f'IP adapters are already installed. Please remove them first.')
            else:
                self.remove_adapters()
        
        # check if ip adapters have data inside them
        for adapter in self.m_ip_adapters.values():
            if not adapter.get_data():
                raise ValueError(f'No data found for {adapter.name} in adapter. Please load data first.')
        
        self._add_adapters_to_model()
        self._register_internal_data()
        
        # remove cache if requested
        if clear_cache:
            for adapter in self.m_ip_adapters.values():
                adapter.clear_cache()
        
    def update_adapters(self) -> None:
        ''' update adapter states, to make the changes effective
        '''
        assert self._retrieve_internal_data() is not None, f'No internal data found for {self.__class__.__name__} in unet. Please install adapters first.'
        weights = [val.weight for key, val in sorted(self.m_ip_adapters.items())]
        self._set_adapter_scale(weights)
        
    def remove_adapters(self) -> None:
        ''' remove ip adapters from unet
        '''
        self._unload_adapters_from_model()
        self._remove_internal_data()

    def _set_adapter_scale(self, scale : np.ndarray | list[float]):
        ''' set ip adapter scale for all ip adapters, following the installation order.
        This is copied from diffusers pipeline
        '''
        
        scale = np.atleast_1d(scale)
        unet = self.m_unet.m_model
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                if len(attn_processor.scale) != len(scale):
                    raise ValueError(
                        f"`scale` should be a list of same length as the number if ip-adapters "
                        f"Expected {len(attn_processor.scale)} but got {len(scale)}."
                    )
                attn_processor.scale = scale
        
    def encode_image_to_embedding(self, images : dict[str, ImageTensor]) -> dict[str, ImagePromptEmbedding]:
        ''' encode image to embedding
        
        parameters
        -------------
        images : dict[str, ImageTensor]
            the input images, in ImageTensor format, the key is the name of the corresponding ip adapter
            
        return
        ---------
        out : dict[str, ImagePromptEmbedding]
            the image prompt embedding for each ip adapter
        '''
        if self.m_image_encoder is None:
            raise ValueError("Image encoder is not set.")
        
        projection_layers : list[ProjectionLayerType] = self.m_unet.m_model.encoder_hid_proj.image_projection_layers
        output = {}
        
        for i, ip_name in enumerate(sorted(self.m_ip_adapters.keys())):
            img_t = images.get(ip_name)
            if img_t is None:
                continue
            
            # get the corresponding projection layer
            proj = projection_layers[i]

            # if the projection layer requires hidden states, then we need to get the hidden states
            # otherwise get embedding directly
            require_hidden_states = not isinstance(proj, df_emb.ImageProjection)
            
            if require_hidden_states:
                enc_positive = self.m_image_encoder.encode(
                    x = img_t.tensor,
                    input_data_layout = img_t.layout,
                    with_hidden_states = True,
                )
                positive_feature = enc_positive.hidden_states[-2]
                
                enc_negative = self.m_image_encoder.encode(
                    x = torch.zeros_like(img_t.tensor),
                    input_data_layout = img_t.layout,
                    with_hidden_states = True,
                )
                negative_feature = enc_negative.hidden_states[-2]
            else:
                enc_positive = self.m_image_encoder.encode(
                    x = img_t.tensor,
                    input_data_layout = img_t.layout,
                    with_hidden_states = False,
                )
                positive_feature = enc_positive.image_embedding
                negative_feature = torch.zeros_like(positive_feature)
                
            img_embeds = ImagePromptEmbedding(
                positive = positive_feature,
                negative = negative_feature,
            )
            
            output[ip_name] = img_embeds
            
        return output
                
            
        
        
        
        