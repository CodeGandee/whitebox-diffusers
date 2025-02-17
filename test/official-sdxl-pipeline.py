# test sdxl pipeline
import numpy as np
import torch

from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel

# convert sdxl safetensors to diffuser
fn_sdxl_model = r'data\models\sdxl\NightVision XL.safetensors'
config_files = {
    'v1':r'misc/sd-configs/v1-inference.yaml',
    'xl':r'misc/sd-configs/sd_xl_base.yaml',
    'v2':r'misc/sd-configs/v2-inference-v.yaml',
    'xl_refiner':r'misc/sd-configs/sd_xl_refiner.yaml'
}
pline = StableDiffusionXLPipeline.from_single_file(fn_sdxl_model, 
                                                   local_files_only=True,
                                                   torch_dtype=torch.float16,
                                                   config_files=config_files)
pline = pline.to('cuda')
images = pline('1girl').images
images[0]