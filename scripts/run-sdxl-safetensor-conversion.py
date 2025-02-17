# test sdxl pipeline
import numpy as np
import torch

USE_SDXL : bool = False
class Options:
    ConvertBase = False
    TestBase = True
    ConvertControlnet = False

if USE_SDXL:
    from diffusers.pipelines import (
        StableDiffusionXLPipeline as StdPipeline,
        StableDiffusionXLControlNetPipeline as StdControlNetPipeline,
    )
else:
    from diffusers.pipelines import (
        StableDiffusionPipeline as StdPipeline,
        StableDiffusionControlNetPipeline as StdControlNetPipeline,
    )

from diffusers.models import ControlNetModel
from diffusers.models import UNet2DConditionModel
import pyaigc.TestHelpers as th
import igpy.myplot.jupyter_plot as jpt


# convert sdxl safetensors to diffuser
# fn_sdxl_model = r'data\models\sdxl\sd_xl_base_1.0.safetensors'
# dir_output_base = r'data\models\sdxl\sdxl-1.0-base-diffusers'

# fn_sdxl_model = r'data\models\sdxl\protovisionXLHighFidelity3D_release0630Bakedvae.safetensors'
# dir_output_base = r'data\models\sdxl\sdxl-protovision-diffusers'

# fn_sdxl_model = r'data\models\sdxl\sdxlYAMERSPERFECTDESIGN_v4COLUMNAARTIS.safetensors'
# dir_output_base = r'data\models\sdxl\sdxl-YamersPerfectDesign-diffusers'

# fn_sdxl_model = r'data\models\sdxl\sdxlUnstableDiffusers_v9DIVINITYMACHINE.safetensors'
# dir_output_base = r'data\models\sdxl\sdxl-UnstableDiffusers-diffusers'

# fn_sdxl_model = r'data\models\sdxl\checkpoint\juggernautXL_v9Rundiffusionphoto2.safetensors'.replace('\\','/')
# dir_output_base = r'data\models\sdxl\checkpoint\sdxl-juggernautXL-diffusers'.replace('\\','/')

fn_model = r'F:\dreamshaper_8.safetensors'.replace('\\','/')
dir_model_output = r'data\models\sd15\checkpoint\dreamshaper-diffuser'.replace('\\','/')

# fn_sdxl_model = r'data\models\sdxl\albedobaseXL_v13.safetensors'
# dir_output_base = r'data\models\sdxl\sdxl-albedobaseXL-diffusers'

dir_sdxl_cnet_canny = r'data\models\sdxl\canny'
fn_test_image = r'sample-data/elijah-cobden-white-tiger-shot-hero.jpg'



# conversion of base model
if Options.ConvertBase:
    import os
    config_files = {
        'v1':r'misc/sd-configs/v1-inference.yaml',
        'xl':r'misc/sd-configs/sd_xl_base.yaml',
        'v2':r'misc/sd-configs/v2-inference-v.yaml',
        'xl_refiner':r'misc/sd-configs/sd_xl_refiner.yaml'
    }
    
    # convert to absolute path
    for k in config_files:
        config_files[k] = os.path.abspath(config_files[k])
    fn_model = os.path.abspath(fn_model)
    
    # now, we need to change cwd to model dependencies
    current_cwd = os.getcwd()
    os.chdir(current_cwd + r'/misc')
    cfg_file = config_files['xl'] if USE_SDXL else config_files['v1']    
    pline = StdPipeline.from_single_file(fn_model, local_files_only=True, original_config_file=cfg_file)
    os.chdir(current_cwd)
    pline.save_pretrained(save_directory=dir_model_output)
    
if Options.TestBase:
    pline = StdPipeline.from_pretrained(dir_model_output)
    pline = pline.to(device = 'cuda', dtype=torch.float16)
    img = pline('snow white',num_inference_steps=20).images[0]
    jpt.imshow(img)
    
if Options.ConvertControlnet:
    torch_device = 'cuda:0'
    model = ControlNetModel.from_pretrained(dir_sdxl_cnet_canny)
    rng = torch.Generator().manual_seed(20)
    
    # try to use it with sdxl
    pline_base = StdPipeline.from_pretrained(dir_model_output)
    sch = pline_base.scheduler
    
    mutable_config = dict(sch.config)
    mutable_config['timestep_spacing'] = 'linspace'
    sch._internal_dict = sch.config.__class__(mutable_config)
    
    if USE_SDXL:
        pline_cnet = StdControlNetPipeline(
            vae=pline_base.vae,
            text_encoder=pline_base.text_encoder,
            text_encoder_2=pline_base.text_encoder_2,
            tokenizer=pline_base.tokenizer,
            tokenizer_2=pline_base.tokenizer_2,
            unet=pline_base.unet,
            controlnet=[model],
            scheduler=sch
        )
    else:
        pline_cnet = StdControlNetPipeline(
            vae=pline_base.vae,
            text_encoder=pline_base.text_encoder,
            tokenizer=pline_base.tokenizer,
            unet=pline_base.unet,
            controlnet=[model],
            scheduler=sch
        )
        
    pline_cnet = pline_cnet.to(device='cuda:0', dtype=torch.float16)
    
    simg = th.SampleImage.from_file(fn_test_image).resize_by_long_edge(long_edge=1024, divisible_by=8)
    edgemap = simg.get_canny_edge()
    edgemap_pt = torch.from_numpy(edgemap).to(dtype=torch.float16, device='cuda')
    edgemap_pt = edgemap_pt[None,None,...].repeat(1,3,1,1) / 255.0
    
    res = pline_cnet(
        prompt="young snow white, beautiful, smile, medieval period, in city, masterpiece, high quality, very sharp, photo realistic, wide angle, unreal engine, single person, good face",
        negative_prompt='blur, low quality, bad face',
        image=[edgemap_pt],
        num_inference_steps=20,
        controlnet_conditioning_scale=0.0,
        generator=rng
    )
    jpt.imshow(res.images[0])