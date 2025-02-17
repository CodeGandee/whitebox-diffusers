import numpy as np
import torch
from attrs import define, field
from rich import print as pprint
import logging
from igpy.common.util_logging import LoggingTemplates
logging.basicConfig(level=logging.INFO, format=LoggingTemplates.NameLevel)

import igpy.common.inout as cio
import igpy.myplot.jupyter_plot as jpt
import igpy.common.shortfunc as sf

import pyaigc.GlobalConfig as C
from pyaigc.pipeline.DiffusionState import (
    ImagePromptData, TextPromptData
)
from test.sd_tester import (
    SDPipeline, SDState, SDTester, PipelineConfig, InputCase
)
from pyaigc.loader.LocalModelLoader import LocalModelLoader
from pyaigc.helper.ip_adapter import (
    IPAdapterHelper, ImagePromptEmbedding
)

USE_SDXL : bool = False

@define(kw_only=True)
class ImagePromptFile:
    name: str = field(default=None)
    path: str = field()
    weight: float = field(default=1.0)
    
    def __attrs_post_init__(self):
        import pathlib
        p = pathlib.Path(self.path).resolve(strict=True)
        
        # use filename as name if not provided
        if self.name is None:
            self.name = p.stem
    
IPFiles : list[ImagePromptFile] = [
    ImagePromptFile(path=r'sample-data\style\0efe48cc-907b-4ee2-824b-7e49d7368bef.png'),
    ImagePromptFile(path=r'sample-data\style\08e7ba55-947d-4663-ad4a-368cc0af83be.png')
]

name2ipf = {x.name: x for x in IPFiles}

ic_base = InputCase(
    positive_prompt='1girl, high quality, masterpiece',
    negative_prompt='low quality, bad quality, blurry',
    num_denoise_step=15,
    image=np.zeros((512,512,3), dtype=np.uint8),
)

# load ip images
for name, ipf in name2ipf.items():
    img = cio.imread(ipf.path)
    ic_base.ip_adapter_images[name] = C.ImageTensor(
        tensor=torch.tensor(img[None,...]),
        layout='nhwc'
    )

# load diffusion model
config = PipelineConfig(
    sd_model_key=C.SDModelKeys.Dreamshaper_sd_v15,
    with_ip_image_encoder=True,
    ip_adapters={
        x.name : C.IPAdapterKeys.sd_v15 for x in IPFiles
    }
)
    
if False:
    mc = C.ModelConfigs.get_instance().get_model_config(config.sd_model_key)

logging.info('loading model pipeline...')
main = SDTester()
_, modelset = main.load_model_pipeline(config)
main.init_input_case(ic_base)
oc = main.init_output_case(ic_base)

# add ip adapter to unet
logging.info('create ip adapter helper...')
ipa_helper = IPAdapterHelper.from_unet(
    main.model_pipeline.m_unet, 
    main.model_pipeline.m_ip_image_encoder,
    modelset.ip_adapters
)

# set weights
for name, ipf in name2ipf.items():
    ipa_helper.m_ip_adapters[name].set_weight(ipf.weight)

# install
logging.info('installing ip adapters...')
ipa_helper.install_adapters(replace_existing=True)
# ip_weight_list = np.linspace(0, 1.0, 12)
w1, w2 = np.meshgrid(np.linspace(0, 0.7, 4), np.linspace(0, 0.5, 4))
ip_weight_list = np.column_stack([w1.flatten(), w2.flatten()])

# embed the image
name2embeds = ipa_helper.encode_image_to_embedding(ic_base.ip_adapter_images)
for name, embed in name2embeds.items():
    oc.dstate.image_prompt_data[name] = ImagePromptData(
        name = name,
        image = ic_base.ip_adapter_images[name],
        positive_embedding= embed.positive,
        negative_embedding= embed.negative        
    )
    
# denoise
import copy
output_image_list : list[np.ndarray]= []
for w in ip_weight_list:
    dstate = copy.deepcopy(oc.dstate)
    for k, ipa in enumerate(ipa_helper.m_ip_adapters.values()):
        ipa.set_weight(w[k])
    ipa_helper.update_adapters()
        
    # dstate.image_prompt_data['0efe48cc-907b-4ee2-824b-7e49d7368bef'].weight = w
    main.denoise_dstate([dstate])
    img = main.model_pipeline.decode_latent_to_image(dstate.latent)[0]
    output_image_list.append(img)

# show
for key, val in ic_base.ip_adapter_images.items():
    jpt.imshow(val.tensor[0].numpy())
    
display_image_list = sf.concat_images_to_groups(output_image_list, 4)
for img in display_image_list:
    jpt.imshow(img)

logging.info('done')

