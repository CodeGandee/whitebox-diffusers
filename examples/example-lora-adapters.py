# test for using adapters in pyaigc pipeline
import numpy as np
import torch
from rich import print as pprint
from attrs import define, field

from igpy.ml.util_diffusers import load_safetensor_as_torch_tensor
import igpy.common.shortfunc as sf
from igpy.common.util_logging import LoggingTemplates, LoggingDateFormat
import igpy.myplot.jupyter_plot as jpt

import logging
logging.basicConfig(level=logging.INFO, 
                    format=LoggingTemplates.TimeLevelMsec, 
                    datefmt=LoggingDateFormat.HourMinSec)

# model loader
from pyaigc.loader.LocalModelLoader import LocalModelLoader
from pyaigc.model.local.LocalLoraAdapter import DiffusionLoraAdapter
import pyaigc.GlobalConfig as C

# adapters
from pyaigc.helper.lora_adapter import LoraAdapterHelper

USE_SDXL : bool = False

AdapterWeightStep = 5

@define(kw_only=True, eq=False)
class LoraConfig:
    name : str = field()
    path : str = field()
    weight_list : np.ndarray = field(factory=lambda: np.linspace(0,1,AdapterWeightStep))

@define(kw_only=True, eq=False)
class TestConfig:
    width : int = 512
    height : int = 512
    positive_prompt : str = '1girl, masterpiece, high quality'
    negative_prompt : str = 'low quality, blur'
    loras : dict[str, LoraConfig] = field(factory=dict)
    random_seed : int | None = 10
    batch_size = 1
    num_denoise_step : int = 20
    
    def __attrs_post_init__(self):
        # check lora key and name consistency
        if self.loras is not None:
            for key, info in self.loras.items():
                if key != info.name:
                    raise ValueError(f'name and key mismatch: {key} != {info.name}')

if USE_SDXL:
    from test.sdxl_tester import (
        SDXLTester as Pipeline,
        SDXLDiffusionState as State,
        InputCase, OutputCase,
        PipelineConfig
    )
    sd_version = C.SDVersion.sdxl
    sd_model_key = C.SDXLModelKeys.Juggernaut
    config  = TestConfig(
        width=512,
        height=512,
        loras={
            'ClassipeintXL' : LoraConfig(
                name='ClassipeintXL',
                path=r'data\models\sdxl\lora\ClassipeintXL2.1.safetensors'
            ),
            'SDXLFaeTastic2400' : LoraConfig(
                name='SDXLFaeTastic2400',
                path=r'data\models\sdxl\lora\SDXLFaeTastic2400.safetensors'
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
    sd_version = C.SDVersion.v15
    sd_model_key = C.SDModelKeys.Dreamshaper_sd_v15    
    config  = TestConfig(
        width=512,
        height=512,
        loras={
            'wrenchsfantasy' : LoraConfig(
                name='wrenchsfantasy',
                path=r'data\models\sd15\lora\wrenchsfantasy-000016.safetensors',
                weight_list=np.linspace(0,2,AdapterWeightStep)
            ),
            'MoXinV1' : LoraConfig(
                name='MoXinV1',
                path=r'data\models\sd15\lora\MoXinV1.safetensors',
                weight_list=np.linspace(0,0.7,AdapterWeightStep)
            ),
        }
    )

    
logging.info(f'loading model pipeline with key: {sd_model_key}')
pc = PipelineConfig(
    sd_model_key=sd_model_key,
    with_canny_control=False,
    with_depth_control=False
)

pipe = Pipeline()
pipe.load_model_pipeline(pc)
# unet_base = pipe.model_pipeline.m_unet.clone().to(device='cpu') # make a backup
# pipe.model_pipeline.m_unet = unet_base.clone().to(device='cuda')

# load lora
name2lora : dict[str, DiffusionLoraAdapter] = {}
for name, info in config.loras.items():
    logging.info(f'loading lora from: {info.path}')
    lora_raw = load_safetensor_as_torch_tensor(info.path)

    # create adapter
    logging.info(f'creating adapter')
    lora_adapter = DiffusionLoraAdapter(
        name=name,
        state_dict=lora_raw.data,
        sd_version=sd_version, 
        metadata=lora_raw.metadata
    )
    
    name2lora[name] = lora_adapter

# load input
ic = InputCase(
    num_denoise_step=config.num_denoise_step,
    image=np.zeros((config.height, config.width, 3), dtype=np.uint8),
    positive_prompt=config.positive_prompt,
    negative_prompt=config.negative_prompt,    
    batch_size=config.batch_size
)

logging.info('installing adapter')
unet = pipe.model_pipeline.m_unet
hlp = LoraAdapterHelper.from_unet(unet)
for name, lora_adapter in name2lora.items():
    hlp.add_adapter(lora_adapter)
    
hlp.install_adapters(replace_existing=True)

weight_for_each_adapter = np.meshgrid(*[info.weight_list for info in config.loras.values()])
name2weights = {}
for i, name in enumerate(config.loras.keys()):
    name2weights[name] = weight_for_each_adapter[i].flatten()
    
n_case = weight_for_each_adapter[0].size
img_per_batch_per_case : list[list[np.ndarray]] = [[] for _ in range(config.batch_size)]
for i in range(n_case):
    # unet = base_unet.clone()
    # pipe.model_pipeline.m_unet = unet
    
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


