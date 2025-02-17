# test for using adapters in pyaigc pipeline
import numpy as np
import torch
from rich import print as pprint
from attrs import define, field

from igpy.ml.util_diffusers import load_safetensor_as_torch_tensor
import igpy.common.shortfunc as sf
import igpy.common.inout as cio
from igpy.common.util_logging import LoggingTemplates, LoggingDateFormat
import igpy.myplot.jupyter_plot as jpt

import logging
logging.basicConfig(level=logging.INFO, 
                    format=LoggingTemplates.TimeLevelMsec, 
                    datefmt=LoggingDateFormat.HourMinSec)

# model loader
from pyaigc.loader.LocalModelLoader import LocalModelLoader
from pyaigc.model.local.LocalIPAdapter import DiffusionIPAdapter
import pyaigc.GlobalConfig as C

# adapters
from pyaigc.helper.ip_adapter import IPAdapterHelper

USE_SDXL : bool = True
USE_CUSTOM_IPA_PATH : bool = True   # use custom ip adapter path
AdapterWeightStep = 4

IPImageFiles : list[str] = [
    r'sample-data\style\0efe48cc-907b-4ee2-824b-7e49d7368bef.png',
    r'sample-data\style\f96fe6cc-e3bd-4e6a-b7aa-6ab76336f6ca.png'
]

@define(kw_only=True, eq=False)
class IPAdapterConfig:
    name : str = field()
    model_key : str = field()
    image : np.ndarray = field()
    weight_list : np.ndarray = field(factory=lambda: np.linspace(0,0.7,AdapterWeightStep))

@define(kw_only=True, eq=False)
class TestConfig:
    width : int = field()
    height : int = field()
    positive_prompt : str = '1girl, masterpiece, high quality'
    negative_prompt : str = 'low quality, blur'
    adapter_configs : dict[str, IPAdapterConfig] = field(factory=dict)
    random_seed : int | None = 10
    batch_size = 1
    num_denoise_step : int = 20
    
    def __attrs_post_init__(self):
        # check lora key and name consistency
        if self.adapter_configs is not None:
            for key, info in self.adapter_configs.items():
                if key != info.name:
                    raise ValueError(f'name and key mismatch: {key} != {info.name}')

if USE_SDXL:
    from test.sdxl_tester import (
        SDXLTester as Pipeline,
        SDXLDiffusionState as State,
        InputCase, OutputCase,
        PipelineConfig
    )
    
    # want to specify the location of ip adapter? use this
    if USE_CUSTOM_IPA_PATH:
        ip_adapter_path = r'data\models\sdxl\ip-adapter\ip_adapter_sdxl\ip_adapter.bin'
        ip_encoder_path = r'data\models\sdxl\ip-adapter\ip_adapter_sdxl_image_encoder'  # a directory containing config.json and model.safetensors
        
        # register the path
        C.ModelConfigs.get_instance().set_base_model_path(C.IPAdapterKeys.sdxl, ip_adapter_path)
        C.ModelConfigs.get_instance().set_base_model_path(C.IPAdapterEncoderKeys.sdxl, ip_encoder_path)
    
    sd_model_key = C.SDXLModelKeys.Albedobase
    sd_version = C.SDVersion.version_from_model_key(sd_model_key)
    model_info = C.ModelConfigs.get_instance().get_model_config(model_key=sd_model_key)
    
    logging.info(f'ip adapter path information')
    pprint(model_info.ip_adapter)
    
    config  = TestConfig(
        width=512,
        height=512,
        adapter_configs={
            'ip_1' : IPAdapterConfig(
                name='ip_1',
                model_key=C.IPAdapterKeys.sdxl,
                image = cio.imread(IPImageFiles[0])
            ),
            'ip_2' : IPAdapterConfig(
                name='ip_2',
                model_key=C.IPAdapterKeys.sdxl,
                image = cio.imread(IPImageFiles[1])
            )
        }
    )
else:
    from test.sd_tester import (
        SDTester as Pipeline,
        SDState as State,
        InputCase, OutputCase, 
        PipelineConfig
    )
    
    # want to specify the location of ip adapter? use this
    if USE_CUSTOM_IPA_PATH:
        ip_adapter_path = r'data\models\sd15\ip-adapter\ip_adapter_sd15\ip_adapter.bin'
        ip_encoder_path = r'data\models\sd15\ip-adapter\ip_adapter_sd_image_encoder'  # a directory containing config.json and model.safetensors
        
        # register the path
        C.ModelConfigs.get_instance().set_base_model_path(C.IPAdapterKeys.sd_v15, ip_adapter_path)
        C.ModelConfigs.get_instance().set_base_model_path(C.IPAdapterEncoderKeys.sd_v15, ip_encoder_path)
    
    sd_model_key = C.SDModelKeys.Dreamshaper_sd_v15    
    sd_version = C.SDVersion.version_from_model_key(sd_model_key)
    model_info = C.ModelConfigs.get_instance().get_model_config(model_key=sd_model_key)
    
    logging.info(f'ip adapter path information')
    pprint(model_info.ip_adapter)
    
    config  = TestConfig(
        width=512,
        height=512,
        adapter_configs={
            'ip_1' : IPAdapterConfig(
                name='ip_1',
                model_key=C.IPAdapterKeys.sd_v15,
                image = cio.imread(IPImageFiles[0])
            ),
            'ip_2' : IPAdapterConfig(
                name='ip_2',
                model_key=C.IPAdapterKeys.sd_v15,
                image = cio.imread(IPImageFiles[1])
            )
        }
    )

    
logging.info(f'loading model pipeline with key: {sd_model_key}')
pc = PipelineConfig(
    sd_model_key=sd_model_key,
    with_ip_image_encoder=True, # this is required to compute image prompt embedding
    with_canny_control=False,
    with_depth_control=False
)

# tell the pipeline to load ip adapters
pc.ip_adapters = {
    name : x.model_key for name, x in config.adapter_configs.items()
}

# load pipeline
pipe = Pipeline()
pipe.load_model_pipeline(pc)

# later you can access the ip adapters through helper
hlp = IPAdapterHelper.from_unet(pipe.model_pipeline.m_unet)
print(f'adapters loaded = {hlp.is_adapter_loaded()}')

# test if the adapters can be installed twice by replacement
if False:
    hlp.install_adapters(replace_existing=True)

# create input
ic = InputCase(
    num_denoise_step=config.num_denoise_step,
    image=np.zeros((config.height, config.width, 3), dtype=np.uint8),
    positive_prompt=config.positive_prompt,
    negative_prompt=config.negative_prompt,    
    batch_size=config.batch_size,
)

# setup ip adapter input
for name, ipa_config in config.adapter_configs.items():
    ic.ip_adapter_images[name] = C.ImageTensor(
        tensor=torch.tensor(ipa_config.image[None,...]),
        layout='nhwc'
    )

weight_for_each_adapter = np.meshgrid(*[info.weight_list for info in config.adapter_configs.values()])
name2weights = {}
for i, name in enumerate(config.adapter_configs.keys()):
    name2weights[name] = weight_for_each_adapter[i].flatten()
    
n_case = weight_for_each_adapter[0].size
img_per_batch_per_case : list[list[np.ndarray]] = [[] for _ in range(config.batch_size)]
for i in range(n_case):
    
    logging.info(f'case {i+1}/{n_case}')
    for ap in hlp.get_adapters().values():
        w = name2weights[ap.name][i]
        ap.set_weight(w)
        logging.info(f'adapter {ap.name} weight: {w}')
        
    logging.info('updating adapters')
    hlp.update_adapters()
    
    logging.info('running diffusion')
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        
    oc = pipe.init_output_case(ic)
    with torch.no_grad():
        pipe.denoise_dstate([oc.dstate])
        imgs = pipe.model_pipeline.decode_latent_to_image(oc.dstate.latent, output_dtype=torch.uint8)
        for k in range(config.batch_size):
            img_per_batch_per_case[k].append(imgs[k])
    
    # print gpu memory usage
    logging.info(f'gpu memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

n_row, n_col = weight_for_each_adapter[0].shape
for ith_batch in range(config.batch_size):
    logging.info(f'batch {ith_batch}')
    img_per_case = img_per_batch_per_case[ith_batch]
    showlist = sf.concat_images_to_groups(img_per_case, n_col)
    for img in showlist:
        jpt.imshow(img)


