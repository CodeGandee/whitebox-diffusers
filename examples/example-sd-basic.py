import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import copy
import numpy as np
from rich import print as pprint
import pyaigc.GlobalConfig as C

C.Logme.get_instance().set_level(C.Logme.Levels.DEBUG)

from test.sdxl_tester import SDXLTester, InputCase, OutputCase, SDTester, PipelineConfig
import igpy.myplot.jupyter_plot as jpt

# setup input files
input_cases = [
    InputCase(
        fn_image=r"sample-data/a1ee6e05-ee42-4883-96c0-21f1d86a5ce4.png",
    ),
    InputCase(
        fn_image=r"sample-data/25c206f1-97bd-4de7-b770-9e8ae16085fd.png",
    ),
]

# use your own model?
USE_CUSTOM_MODEL: bool = False

# use sdxl for sd 1.5?
USE_SDXL: bool = True

# for controlnet,
# use local annotator or remote annotator (requires download)
USE_LOCAL_ANNOTATOR = True

# controlnets
# if enabled, the controlnet model must be available, but you can choose not to use them
WITH_CANNY_CONTROL: bool = False
WITH_DEPTH_CONTROL: bool = False
WITH_OPENPOSE_CONTROL: bool = True

# output image size's longer edge length
OUTPUT_IMAGE_LONG_EDGE_LENGTH: int = 640
DEFAULT_DENOISE_STEP: int = 10
DEFAULT_BATCH_SIZE: int = 2


# controlnet weights, if used
class ControlnetWeights:
    Canny = 0.5
    Depth = 0.3
    Openpose = 0.7


if USE_SDXL:
    from test.sdxl_tester import SDXLTester, InputCase, OutputCase
else:
    from test.sd_tester import SDTester, InputCase, OutputCase

# select model key, or specify custom model locations
if USE_SDXL:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_key = C.SDXLModelKeys.CustomModel_1
        dir_sdxl_model = r"data/models/sdxl/sdxl-JuggernautXL-v5-diffusers"
        dir_canny_model = r"data/models/sdxl/canny"

        C.ModelConfigs.get_instance().set_base_model_path(sd_key, dir_sdxl_model)
        C.ModelConfigs.get_instance().set_base_model_path(
            C.ControlNetModelKeys.Canny_sdxl, dir_canny_model
        )
    else:
        # use default model paths
        sd_key = C.SDXLModelKeys.ProtoVision
else:
    if USE_CUSTOM_MODEL:
        # use a model on your disk
        sd_key = C.SDModelKeys.CustomModel_1
        dir_sd_model = r"data/models/diffusion/openjourney"
        dir_canny_model = r"data/models/controlnet/sd-controlnet-canny"

        C.ModelConfigs.get_instance().set_base_model_path(sd_key, dir_sd_model)
        C.ModelConfigs.get_instance().set_base_model_path(
            C.ControlNetModelKeys.Canny_sd_v15, dir_canny_model
        )
    else:
        # use default model paths
        sd_key = C.SDModelKeys.Dreamshaper_sd_v15

# where is the model?
model_info = C.ModelConfigs.get_instance().get_model_config(sd_key)
print("--- model components ---")
pprint(model_info)
print("------------------------")

# load model, create pipeline
if USE_SDXL:
    sdtest = SDXLTester()
else:
    sdtest = SDTester()

pc = PipelineConfig(
    sd_model_key=sd_key,
    with_canny_control=WITH_CANNY_CONTROL,
    with_depth_control=WITH_DEPTH_CONTROL,
    with_openpose_control=WITH_OPENPOSE_CONTROL,
)
sdtest.load_model_pipeline(pc)

ic_template = copy.deepcopy(input_cases[1])
ic_template.batch_size = DEFAULT_BATCH_SIZE
ic_template.num_denoise_step = DEFAULT_DENOISE_STEP
ic_template.positive_prompt = "attractive young girl"
ic_template.negative_prompt = "toy, blur, naked"

if False:
    # case: no control, text-to-image only
    print("case: no control, text-to-image only")
    ic = copy.deepcopy(ic_template)
    sdtest.init_input_case(ic)  # init input case

    # resize to the long edge length
    ic.resize_by_long_edge(OUTPUT_IMAGE_LONG_EDGE_LENGTH, divisible_by=8)
    oc = sdtest.init_output_case(ic)  # init output case

    # denoise to the end
    sdtest.denoise_dstate([oc.dstate])
    img_batch = sdtest.model_pipeline.decode_latent_to_image(
        oc.dstate.latent, output_dtype=torch.uint8
    )

    # show the source image along with the generated image
    print("generated image")
    imglist = [ic.image] + [x for x in img_batch]
    imgshow = np.column_stack(imglist)
    jpt.imshow(imgshow)

if False:
    # case: no control, image-to-image
    print("case: no control, image-to-image")

    # set denoise to time step
    # img-to-img noise sequence
    noise_endpoints = np.logspace(np.log10(600), np.log10(900), num=3, dtype=int)
    imglist_per_denoise = [[] for _ in range(ic_template.batch_size)]
    for k, noise_pt in enumerate(noise_endpoints):
        ic = copy.deepcopy(ic_template)
        ic.init_noise_timestep_index = noise_pt
        ic.num_denoise_step = DEFAULT_DENOISE_STEP
        sdtest.init_input_case(ic)

        # resize to the long edge length
        ic.resize_by_long_edge(OUTPUT_IMAGE_LONG_EDGE_LENGTH, divisible_by=8)
        oc = sdtest.init_output_case(ic)

        # denoise to the end
        sdtest.denoise_dstate([oc.dstate])
        img_batch = sdtest.model_pipeline.decode_latent_to_image(
            oc.dstate.latent, output_dtype=torch.uint8, force_upcast=False
        )

        # add the generated image to the list
        for i in range(ic.batch_size):
            if k == 0:
                imglist_per_denoise[i].append(ic.image)
            imglist_per_denoise[i].append(img_batch[i])

    # show the generated images
    for i in range(ic.batch_size):
        imgshow = np.column_stack(imglist_per_denoise[i])
        jpt.imshow(imgshow)

if True:
    # case: control, text-to-image
    print("case: control, text-to-image")

    from controlnet_aux import (
        OpenposeDetector,
        ZoeDetector,
        CannyDetector,
    )

    from easy_dwpose import DWposeDetector

    for control_type in ["canny", "depth", "pose"]:
        ic = copy.deepcopy(ic_template)
        sdtest.init_input_case(ic)  # init input case

        control_image = None
        if control_type == "canny" and WITH_CANNY_CONTROL:
            ic.canny_weight = ControlnetWeights.Canny
            canny_annotator = CannyDetector()
            ic.canny_image = np.array(canny_annotator(ic.image))
        elif control_type == "depth" and WITH_DEPTH_CONTROL:
            ic.depth_weight = ControlnetWeights.Depth
            if USE_LOCAL_ANNOTATOR:
                depth_annotator = ZoeDetector.from_pretrained("data/models/Annotators")
            else:
                depth_annotator = ZoeDetector.from_pretrained("lllyasviel/Annotators")
            ic.depth_image = np.array(depth_annotator(ic.image))
        elif control_type == "pose" and WITH_OPENPOSE_CONTROL:
            ic.openpose_weight = ControlnetWeights.Openpose
            print("using dwpose for pose control")
            openpose_annotator = DWposeDetector(device=C.DEFAULT_TORCH_DEVICE)
            # if USE_LOCAL_ANNOTATOR:
            #     openpose_annotator = OpenposeDetector.from_pretrained(
            #         "data/models/Annotators"
            #     )
            # else:
            #     openpose_annotator = OpenposeDetector.from_pretrained(
            #         "lllyasviel/Annotators"
            #     )
            ic.openpose_image = np.array(openpose_annotator(ic.image))
        else:
            print(f"control type {control_type} is not enabled")
            continue

        # resize to the long edge length
        ic.resize_by_long_edge(OUTPUT_IMAGE_LONG_EDGE_LENGTH, divisible_by=8)
        oc = sdtest.init_output_case(ic)  # init output case

        # denoise to the end
        sdtest.denoise_dstate([oc.dstate])
        img_batch = sdtest.model_pipeline.decode_latent_to_image(
            oc.dstate.latent, output_dtype=torch.uint8
        )

        # show the source image along with the generated image
        print(f"generated image with control: {control_type}")
        control_image = None
        if control_type == "canny":
            control_image = ic.canny_image
        elif control_type == "depth":
            control_image = ic.depth_image
        elif control_type == "pose":
            control_image = ic.openpose_image

        imglist = [ic.image] + [control_image] + [x for x in img_batch]
        imgshow = np.column_stack(imglist)
        jpt.imshow(imgshow)
