import numpy as np
import torch
import unittest
from rich import print as pprint
from attrs import define, field, asdict

import pyaigc.GlobalConfig as C
from pyaigc.pipeline.StableDiffusionWithControl import DiffusionState as SDState, StableDiffusionWithControl as SDPipeline
from .sd_tester import (
    SDTester,
    ControlnetData, ControlNetInfo,
    InputCase, OutputCase,
    PipelineConfig,
    DiffusionModelBundle
)

from pyaigc.pipeline.SDXLWithControl import (
    SDXLWithControl,
    SDXLDiffusionState,
)

import pyaigc.TestHelpers as th

@define(kw_only=True, eq=False)
class SDXLTester(SDTester):    
    
    def _create_pipeline(self) -> SDXLWithControl:
        return SDXLWithControl()

    def load_model_pipeline(self, config : PipelineConfig) -> tuple[SDPipeline, DiffusionModelBundle]:
        
        obj : SDXLWithControl = None
        obj, mds = super().load_model_pipeline(config)
        obj.m_text_model_2 = mds.text_model_2
        return obj, mds
    
    def _create_diffusion_state(self, target_width: int, target_height: int) -> SDXLDiffusionState:
        return SDXLDiffusionState(target_width=target_width, target_height=target_height)