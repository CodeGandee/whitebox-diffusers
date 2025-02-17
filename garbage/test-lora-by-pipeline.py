# test sdxl pipeline
import numpy as np
import torch
from rich import print as pprint
from diffusers.models import UNet2DConditionModel
import copy

USE_SDXL = False

if USE_SDXL:
    from diffusers.pipelines.auto_pipeline import (
        AutoPipelineForText2Image as StdPipeline,
        # StableDiffusionXLPipeline as StdPipeline,
        StableDiffusionXLControlNetPipeline as StdControlNetPipeline,
    )
    from diffusers.loaders.lora import StableDiffusionXLLoraLoaderMixin as lora_helper
else:
    from diffusers.pipelines.auto_pipeline import (
        StableDiffusionPipeline as StdPipeline,
        AutoPipelineForText2Image as StdPipeline,
        StableDiffusionControlNetPipeline as StdControlNetPipeline,
    )
    from diffusers.loaders.lora import LoraLoaderMixin as lora_helper

from diffusers.models import UNet2DConditionModel

import pyaigc.TestHelpers as th
import igpy.myplot.jupyter_plot as jpt
from igpy.ml.util_diffusers import (
    load_safetensor_as_torch_tensor,
    AdapterInfo, AdapterType, StableDiffusionAdapterHelper
)
from igpy.common.shortfunc import concat_images_to_groups

# convert sdxl safetensors to diffuser
# dir_sdxl_model = r'data\models\sdxl\checkpoint\sdxl-protovision-diffusers'.replace('\\','/')
# fn_lora = r'data\models\sdxl\lora\xl_more_art-full_v1.safetensors'

dir_sd_model = r'data\models\sd15\checkpoint\dreamshaper-diffuser'
fn_lora = r'data\models\sd15\lora\MoXinV1.safetensors'

# lora_data, lora_meta = load_safetensor_as_torch_tensor(fn_lora)

pline = StdPipeline.from_pretrained(dir_sd_model)
unet : UNet2DConditionModel = pline.unet

hlp = StableDiffusionAdapterHelper(unet=unet)

lora_raw = load_safetensor_as_torch_tensor(fn_lora)
# lora_dict, network_alphas = lora_helper.lora_state_dict(
#     lora_raw.data, unet_config=unet.config)

hlp.load_lora_by_state_dict('mylora_1', lora_raw.data, weight=0)
hlp.load_lora_by_state_dict('mylora_2', lora_raw.data, weight=0)

hlp.update_adapters()
# pprint(lora_dict.keys())

# assert False

# lora_helper.load_lora_into_unet(lora_dict, network_alphas, unet, adapter_name='mylora')

# pline.load_lora_weights(fn_lora, adapter_name='mylora')
# pline.set_adapters(['mylora'])
# pline.load_lora_weights(fn_lora)
# pline.unet.load_attn_procs(fn_lora)
pline = pline.to(device = 'cuda', dtype=torch.float16)

lora_weights = np.linspace(0,0.7,6)
n_infer_step = 20
width, height = 512, 512
seed_num : int = 100
prompt = 'young lovely snow white, long hair, realistic, portrait, color photo'
negative_prompt = 'bad quality'

imglist = []
for i, w in enumerate(lora_weights):
    g = torch.random.manual_seed(seed_num)
    
    if 'mylora_1' in hlp.adapters:
        hlp.adapters['mylora_1'].weight = w
    if 'mylora_2' in hlp.adapters:
        hlp.adapters['mylora_2'].weight = lora_weights.max() - w
        
    if i % 2 == 0:
        hlp.adapters['mylora_2'].is_enabled = False
    else:
        hlp.adapters['mylora_2'].is_enabled = True
        
    hlp.update_adapters()
    
    # pline.set_adapters(['mylora'],[w])
    
        
    # pline = pline.to(device = 'cuda', dtype=torch.float16)
    # img = pline(prompt,num_inference_steps=n_infer_step, width=width, height=height, generator=g,
    #             cross_attention_kwargs={'scale':w}).images[0]
    img = pline(prompt, negative_prompt=negative_prompt,
                num_inference_steps=n_infer_step, width=width, height=height, generator=g).images[0]
    imglist.append(img)
    
imgshowlist = concat_images_to_groups(imglist, 3)
for x in imgshowlist:
    jpt.imshow(x)
    
# g = torch.random.manual_seed(100)
# img = pline(prompt,num_inference_steps=n_infer_step, width=512, height=800, generator=g,
#             cross_attention_kwargs={'scale':0.0}).images[0]

# g = torch.random.manual_seed(100)
# img_2 = pline(prompt,num_inference_steps=20, width=512, height=800, generator=g,
#               cross_attention_kwargs={'scale':0.1}).images[0]
# jpt.imshow(np.column_stack((img,img_2)))