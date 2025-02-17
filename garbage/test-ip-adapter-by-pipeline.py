# test sdxl pipeline
import numpy as np
import torch
from rich import print as pprint
import copy
import PIL.Image as pimg
import igpy.myplot.jupyter_plot as jpt
from attrs import define, field

import igpy.common.inout as cio
from diffusers.models import UNet2DConditionModel
from transformers.models.clip import CLIPVisionModelWithProjection
from diffusers.loaders.ip_adapter import IPAdapterMixin

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
        # StableDiffusionPipeline as StdPipeline,
        AutoPipelineForText2Image as StdPipeline,
        StableDiffusionControlNetPipeline as StdControlNetPipeline,
        StableDiffusionPipeline as StdSimplePipeline,
    )
    from diffusers.loaders.lora import LoraLoaderMixin as lora_helper

from diffusers.models import (
    UNet2DConditionModel,
)


from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

import pyaigc.TestHelpers as th
import igpy.myplot.jupyter_plot as jpt
from igpy.ml.util_diffusers import (
    load_safetensor_as_torch_tensor,
    AdapterInfo, AdapterType, StableDiffusionAdapterHelper,
)
from igpy.common.shortfunc import concat_images_to_groups

# convert sdxl safetensors to diffuser
# dir_sdxl_model = r'data\models\sdxl\checkpoint\sdxl-protovision-diffusers'.replace('\\','/')
# fn_lora = r'data\models\sdxl\lora\xl_more_art-full_v1.safetensors'
@define(kw_only=True)
class InputCase:
    width : int = field(default=512)
    height : int = field(default=512)
    num_inference_steps : int = field(default=50)
    positive_prompt : str = field(default='1girl')
    negative_prompt : str = field(default='monochrome, lowres, bad anatomy, worst quality, low quality')
    
    fn_ipa_image : str | None = field(default=None) # ip adapter input image file
    ipa_image : np.ndarray = field(default=None)    # ip adapter input image
    ipa_embeds_positive : torch.Tensor = field(default=None) # ip adapter input positive embeds
    ipa_embeds_negative : torch.Tensor = field(default=None) # ip adapter input negative embeds
    
    def load_image(self):
        if self.fn_ipa_image is not None:
            self.ipa_image = cio.imread(self.fn_ipa_image)
            
    def get_ipa_image_tensor(self) -> torch.Tensor:
        ''' get tensor in NCHW format, in float32
        '''
        img = self.ipa_image
        out : torch.Tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
        return out
    
    def get_ipa_image_pil(self) -> pimg.Image:
        ''' get PIL image
        '''
        img = self.ipa_image
        out = pimg.fromarray(img)
        return out
        
    
# load image
ic = InputCase()
ic.fn_ipa_image = r'sample-data\b6184bf1-cd84-4d70-82a5-f0f3d7ecfa13.png'
ic.load_image()

# load models
dir_sd_model = r'data\models\sd15\checkpoint\dreamshaper-diffuser'
fn_ipa_encoder = r'data\models\sd15\ip-adapter\ip_adapter_sd_image_encoder'
fn_ipa_std = r'data\models\sd15\ip-adapter\ip_adapter_sd15\ip_adapter.bin'

# load encoder
img_encoder = CLIPVisionModelWithProjection.from_pretrained(fn_ipa_encoder)
# img_encoder = None

# load pipeline
n_ip_adapter = 3
pline = StdPipeline.from_pretrained(dir_sd_model, safety_checker = None, image_encoder=img_encoder)
pline.scheduler = DPMSolverMultistepScheduler.from_config(pline.scheduler.config)
pline.load_ip_adapter([fn_ipa_std]*n_ip_adapter, subfolder=['']*n_ip_adapter, weight_name=['']*n_ip_adapter)
pline.to(device = 'cuda', dtype=torch.float16)
pline.set_ip_adapter_scale(0.2)

# convert image to tensor
img_done = pline(
    width=ic.width, height=ic.height,
    prompt=[ic.positive_prompt],
    negative_prompt=[ic.negative_prompt],
    ip_adapter_image=[ic.get_ipa_image_pil()]*n_ip_adapter,
    num_inference_steps=ic.num_inference_steps,
    num_images_per_prompt = 3
)
img_output = img_done.images[0]
jpt.imshow(img_output)

