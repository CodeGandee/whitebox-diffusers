import pathlib
import numpy as np
import torch
from typing import Union, Callable, ClassVar
import logging
import os

from igpy.common.util_logging import add_logging_level, CustomLogger
from igpy.util_torch.helpers import (
    TorchDevice,
    get_all_torch_devices,
    get_canonical_torch_device,
)
from igpy.common.EnumBase import CustomEnumBase
from igpy.ml.util_diffusers import AdapterType, SafetensorData

from attrs import define, field, fields, Attribute
import attrs.validators as av

NUM_MAX_USER = 10
NUM_MAX_SESSION_PER_USER = 3
SESSION_CACHE_TIMEOUT_SEC = 60 * 60 * 1 * 1  # 1 hour

if torch.cuda.is_available():
    DEFAULT_TORCH_DTYPE = torch.float16
    DEFAULT_TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEFAULT_TORCH_DTYPE = torch.float16
    DEFAULT_TORCH_DEVICE = torch.device("mps")
else:
    DEFAULT_TORCH_DTYPE = torch.float32  # torch.float16
    DEFAULT_TORCH_DEVICE = torch.device("cpu")  # torch.device('cuda:0')


@define(kw_only=True, eq=False, slots=False)
class ImageTensor:
    tensor: torch.Tensor = field()
    layout: str = field(default="nchw")

    @tensor.validator
    def _check_tensor(self, attribute, value: torch.Tensor | None):
        if value is None:
            return

        # must be 4d tensor
        assert value.ndim == 4, f"expected 4D tensor, got {value.ndim}D"

        # dtype must be uint8, float16, float32
        assert value.dtype in (
            torch.uint8,
            torch.float16,
            torch.float32,
        ), f"unsupported dtype {value.dtype}"

    @layout.validator
    def _check_layout(self, attribute, value: str):
        assert value is not None, "layout must be specified"
        assert value in (
            "nchw",
            "nhwc",
        ), f"unsupported layout {value}, must be nchw or nhwc"


class LoggingLevel(CustomEnumBase):
    Debug = logging.DEBUG + 1
    Info = logging.INFO + 1
    Warning = logging.WARNING + 1
    Error = logging.ERROR + 1


class LoggingTag(CustomEnumBase):
    Debug = "AIGC_DEBUG"
    Info = "AIGC_INFO"
    Warning = "AIGC_WARNING"
    Error = "AIGC_ERROR"


# create loggers
add_logging_level(LoggingTag.Debug, LoggingLevel.Debug, LoggingTag.Debug.lower())
add_logging_level(LoggingTag.Info, LoggingLevel.Info, LoggingTag.Info.lower())
add_logging_level(LoggingTag.Warning, LoggingLevel.Warning, LoggingTag.Warning.lower())
add_logging_level(LoggingTag.Error, logging.ERROR, LoggingTag.Error.lower())

# loggers, you need to call setLevel(LoggingLevel.xxx) first to enable the log level
log_debug: Callable[[str], None] = logging.getLogger().aigc_debug
log_info: Callable[[str], None] = logging.getLogger().aigc_info
log_warning: Callable[[str], None] = logging.getLogger().aigc_warning
log_error: Callable[[str], None] = logging.getLogger().aigc_error


def set_log_level(level: int):
    logging.basicConfig(level=level)


class SDVersion(CustomEnumBase):
    v15 = "sd-base-1.5"
    sdxl = "sdxl"

    @classmethod
    def version_from_model_key(cls, model_key: str) -> str:
        """get the version of the model from the model key.
        If the model key is not recognized, return None.
        """

        # FIXME: assuming model key naming rule : primary_model_key + '.' + component_key
        # if this rule is changed, this function must be updated
        primary_model_key: str = model_key.split(".")[0]

        if primary_model_key in SDModelKeys.get_enum_values():
            return cls.v15
        elif primary_model_key in SDXLModelKeys.get_enum_values():
            return cls.sdxl
        else:
            return None


class SDModelKeys(CustomEnumBase):
    StableDiffusion_v15 = "StableDiffusion_v15"
    Dreamshaper_sd_v15 = "Dreamshaper_sd_v15"
    Openjourney_sd_v15 = "Openjourney_sd_v15"

    # these are for custom models, you can use them to load your own models
    CustomModel_1 = "SD_CustomModel_1"
    CustomModel_2 = "SD_CustomModel_2"
    CustomModel_3 = "SD_CustomModel_3"
    CustomModel_4 = "SD_CustomModel_4"
    CustomModel_5 = "SD_CustomModel_5"

    @classmethod
    def get_base_model(cls) -> str:
        """return the primary model key, which is the default model key of this kind of sd"""
        return cls.Dreamshaper_sd_v15


class SDXLModelKeys(CustomEnumBase):
    SDXL_base_v1 = "SDXL_v1"
    ProtoVision = "SDXL_ProtoVision"
    NightVision = "SDXL_NightVision"
    Juggernaut = "SDXL_Juggernaut"
    UnstableDiffusers = "SDXL_UnstableDiffusers"
    YamersPerfectDesign = "SDXL_YamersPerfectDesign"
    Albedobase = "SDXL_Albedobase"

    # these are for custom models, you can use them to load your own models
    CustomModel_1 = "SDXL_CustomModel_1"
    CustomModel_2 = "SDXL_CustomModel_2"
    CustomModel_3 = "SDXL_CustomModel_3"
    CustomModel_4 = "SDXL_CustomModel_4"
    CustomModel_5 = "SDXL_CustomModel_5"

    @classmethod
    def get_base_model(cls) -> str:
        """return the primary model key, which is the default model key of this kind of sd"""
        return cls.SDXL_base_v1


# maximum possible value of timestep in sd 1.5
MaxTimestep_sd_v15 = 1000 - 1


class ControlNetModelKeys(CustomEnumBase):
    Depth_sd_v15 = "ControlNet_Depth_sd_v15"
    Canny_sd_v15 = "ControlNet_Canny_sd_v15"
    OpenPose_sd_v15 = "ControlNet_OpenPose_sd_v15"
    Depth_sdxl = "ControlNet_Depth_sdxl"
    Canny_sdxl = "ControlNet_Canny_sdxl"
    OpenPose_sdxl = "ControlNet_OpenPose_sdxl"

    # for custom models
    CustomControlNet_1 = "CustomControlNet_1"
    CustomControlNet_2 = "CustomControlNet_2"
    CustomControlNet_3 = "CustomControlNet_3"
    CustomControlNet_4 = "CustomControlNet_4"
    CustomControlNet_5 = "CustomControlNet_5"


class TextModelKeys(CustomEnumBase):
    CLIP_TextEncoder = "CLIP_TextEncoder"
    CLIP_TextTokenizer = "CLIP_TextTokenizer"

    CLIP_TextModel_prompt_1 = "CLIP_Text_prompt_1"
    CLIP_TextEncoder_prompt_1 = "CLIP_Text_prompt_1_enc"  # for prompt 1
    CLIP_TextTokenizer_prompt_1 = "CLIP_Text_prompt_1_tok"  # for prompt 1

    CLIP_TextModel_prompt_2 = "CLIP_Text_prompt_2"
    CLIP_TextEncoder_prompt_2 = "CLIP_Text_prompt_2_enc"  # for prompt 2
    CLIP_TextTokenizer_prompt_2 = "CLIP_Text_prompt_2_tok"  # for prompt 2


class IPAdapterEncoderKeys(CustomEnumBase):
    sdxl = "IPAdapterEncoder_sdxl"
    sd_v15 = "IPAdapterEncoder_sd_v15"


class IPAdapterKeys(CustomEnumBase):
    sdxl = "IPAdapter_sdxl"
    sd_v15 = "IPAdapter_sd_v15"


class ImageProcessorKeys(CustomEnumBase):
    CLIP_ImageProcessor = "CLIP_ImageProcessor"


class SchedulerTypes(CustomEnumBase):
    """types of the schedulers, see https://huggingface.co/docs/diffusers/api/schedulers/overview for detail"""

    DPM_2M_Karras = "DPM_2M_Karras"
    DPM_2M = "DPM_2M"
    DPM_2M_SDE = "DPM_2M_SDE"  # DO NOT USE
    DPM_2M_SDE_Karras = "DPM_2M_SDE_Karras"  # DO NOT USE
    # UniPCMS = 'UniPCMultistep' #DO NOT USE
    EulerDiscrete = "EulerDiscrete"

    @classmethod
    def get_prefer_order(cls) -> list[str]:
        """return a list of scheduler types, in the order of preference."""
        return [
            cls.EulerDiscrete,
            cls.DPM_2M_Karras,
            cls.DPM_2M,
            cls.DPM_2M_SDE,
            cls.DPM_2M_SDE_Karras,
        ]


@define(kw_only=True, eq=False)
class ModelComponent:
    key: str = field(validator=av.instance_of(str))
    path: str | None = field(default=None, validator=av.optional(av.instance_of(str)))

    # when loading the model, you can specify the dtype and device here
    torch_dtype: torch.dtype = field()
    torch_device: torch.device | str = field()

    # must be defined this way because DEFAULT_TORCH_DTYPE and DEFAULT_TORCH_DEVICE might be changed during runtime
    @torch_dtype.default
    def _default_torch_dtype(self):
        return DEFAULT_TORCH_DTYPE

    @torch_device.default
    def _default_torch_device(self):
        return DEFAULT_TORCH_DEVICE


@define(kw_only=True, eq=False)
class SchedulerComponent(ModelComponent):
    # type of the scheduler when loaded
    scheduler_type: str = field(validator=av.instance_of(str))


@define(kw_only=True, eq=False)
class TextModelComponent(ModelComponent):
    """component of the text model, including the encoder and tokenizer.
    The path of the text model does not matter, because component paths are given
    in individual component config.
    """

    encoder: ModelComponent = field()
    tokenizer: ModelComponent = field()


@define(kw_only=True)
class ControlnetModelConfig:
    canny: ModelComponent | None = field(default=None)
    depth: ModelComponent | None = field(default=None)
    openpose: ModelComponent | None = field(default=None)

    def set_torch_dtype(self, torch_dtype: torch.dtype):
        if self.canny is not None:
            self.canny.torch_dtype = torch_dtype
        if self.depth is not None:
            self.depth.torch_dtype = torch_dtype
        if self.openpose is not None:
            self.openpose.torch_dtype = torch_dtype

    def set_torch_device(self, torch_device: torch.device | str):
        if self.canny is not None:
            self.canny.torch_device = torch_device
        if self.depth is not None:
            self.depth.torch_device = torch_device
        if self.openpose is not None:
            self.openpose.torch_device = torch_device


@define(kw_only=True)
class IPAdapterModelConfig:
    encoder: ModelComponent | None = field(default=None)

    # name => adapter, by default, the name is the model key of the adapter
    adapters: dict[str, ModelComponent] = field(factory=dict)

    def set_torch_dtype(self, torch_dtype: torch.dtype):
        if self.encoder is not None:
            self.encoder.torch_dtype = torch_dtype
        if self.adapters is not None:
            for k, v in self.adapters.items():
                v.torch_dtype = torch_dtype

    def set_torch_device(self, torch_device: torch.device | str):
        if self.encoder is not None:
            self.encoder.torch_device = torch_device
        if self.adapters is not None:
            for k, v in self.adapters.items():
                v.torch_device = torch_device

    def rename_adapters(self, name2key: dict[str, str]):
        """set the names of the adapters, mapping the name to the key."""
        key2adapter = self.adapters
        self.adapters = {}

        import copy

        for name, key in name2key.items():
            self.adapters[name] = copy.deepcopy(key2adapter[key])


@define(kw_only=True)
class SDModelConfig:
    primary_model_key: str = field(validator=av.instance_of(str))
    text_model: TextModelComponent = field()

    # only available in SDXL
    text_model_2: TextModelComponent | None = field(default=None)

    unet: ModelComponent = field()
    vae: ModelComponent = field()
    scheduler: SchedulerComponent = field()
    image_processor: ModelComponent | None = field(default=None)  # used by clip model

    # controlnets
    controlnets: ControlnetModelConfig = field(factory=ControlnetModelConfig)

    # ip adapter config
    ip_adapter: IPAdapterModelConfig | None = field(default=None)

    def is_valid(self) -> bool:
        return self.unet is not None and self.unet.path is not None

    def get_sd_version(self) -> str:
        if self.primary_model_key is None:
            return None

        return SDVersion.version_from_model_key(self.primary_model_key)

    @primary_model_key.validator
    def _check_primary_model_key(self, attribute, value):
        assert isinstance(
            value, str
        ), f"primary_model_key must be str, got {type(value)}"
        is_sd_15: bool = value in SDModelKeys.get_enum_values()
        is_sdxl: bool = value in SDXLModelKeys.get_enum_values()
        assert (
            is_sd_15 or is_sdxl
        ), f"primary_model_key must be one of {SDModelKeys.get_enum_values()} or {SDXLModelKeys.get_enum_values()}, got {value}"

    def set_torch_dtype(self, torch_dtype: torch.dtype):
        """set all dtype of all components to the given dtype"""
        a: Attribute = None
        for a in fields(self.__class__):
            if isinstance(a.type, TextModelComponent):
                obj: TextModelComponent = getattr(self, a.name)
                if obj.encoder is not None:
                    obj.encoder.torch_dtype = torch_dtype

                if obj.tokenizer is not None:
                    obj.tokenizer.torch_dtype = torch_dtype

            elif isinstance(a.type, ModelComponent):
                obj: ModelComponent = getattr(self, a.name)
                obj.torch_dtype = torch_dtype

            elif isinstance(a.type, ControlnetModelConfig):
                cnet_obj: ControlnetModelConfig = getattr(self, a.name)
                cnet_obj.set_torch_dtype(torch_dtype)

            elif isinstance(a.type, IPAdapterModelConfig):
                ip_obj: IPAdapterModelConfig = getattr(self, a.name)
                ip_obj.set_torch_dtype(torch_dtype)

    def set_torch_device(self, torch_device: torch.device | str):
        """set all device of all components to the given device"""

        a: Attribute = None
        for a in fields(self.__class__):
            if isinstance(a.type, TextModelComponent):
                obj: TextModelComponent = getattr(self, a.name)
                if obj.encoder is not None:
                    obj.encoder.torch_device = torch_device

                if obj.tokenizer is not None:
                    obj.tokenizer.torch_device = torch_device

            elif isinstance(a.type, ModelComponent):
                obj: ModelComponent = getattr(self, a.name)
                obj.torch_device = torch_device

            elif isinstance(a.type, ControlnetModelConfig):
                cnet_obj: ControlnetModelConfig = getattr(self, a.name)
                cnet_obj.set_torch_device(torch_device)

            elif isinstance(a.type, IPAdapterModelConfig):
                ip_obj: IPAdapterModelConfig = getattr(self, a.name)
                ip_obj.set_torch_device(torch_device)


# singleton of model configs
_instance_ModelConfigs: "ModelConfigs" = None


@define(kw_only=True, eq=False)
class ModelConfigs:
    base_model_paths: dict[str, str] = field(factory=dict)
    use_shared_model_components: bool = field(default=False)
    DefaultSchedulerType_sd_v15: ClassVar[str] = field(
        default=SchedulerTypes.DPM_2M_Karras
    )
    DefaultSchedulerType_sdxl: ClassVar[str] = field(
        default=SchedulerTypes.EulerDiscrete
    )

    @classmethod
    def get_instance(cls) -> "ModelConfigs":
        global _instance_ModelConfigs

        if _instance_ModelConfigs is None:
            _instance_ModelConfigs = cls()
            _instance_ModelConfigs._init_default_paths()

        return _instance_ModelConfigs

    def _init_default_paths(self):
        basedir = self.get_default_model_root()
        self.base_model_paths = {
            SDModelKeys.StableDiffusion_v15: f"{basedir}/sd15/checkpoint/stable-diffusion-v1-5",
            SDModelKeys.Dreamshaper_sd_v15: f"{basedir}/sd15/checkpoint/dreamshaper-diffuser",
            SDModelKeys.Openjourney_sd_v15: f"{basedir}/sd15/checkpoint/openjourney",
            SDXLModelKeys.SDXL_base_v1: f"{basedir}/sdxl/checkpoint/sdxl-1.0-base-diffusers",
            SDXLModelKeys.NightVision: f"{basedir}/sdxl/checkpoint/sdxl-NightVision-diffusers",
            SDXLModelKeys.ProtoVision: f"{basedir}/sdxl/checkpoint/sdxl-protovision-diffusers",
            SDXLModelKeys.Juggernaut: f"{basedir}/sdxl/checkpoint/sdxl-JuggernautXL-diffusers",
            SDXLModelKeys.UnstableDiffusers: f"{basedir}/sdxl/checkpoint/sdxl-UnstableDiffusers-diffusers",
            SDXLModelKeys.YamersPerfectDesign: f"{basedir}/sdxl/checkpoint/sdxl-YamersPerfectDesign-diffusers",
            SDXLModelKeys.Albedobase: f"{basedir}/sdxl/checkpoint/sdxl-albedobaseXL-diffusers",
            ControlNetModelKeys.Canny_sd_v15: f"{basedir}/sd15/controlnet/sd-controlnet-canny",
            ControlNetModelKeys.Depth_sd_v15: f"{basedir}/sd15/controlnet/sd-controlnet-depth",
            ControlNetModelKeys.OpenPose_sd_v15: f"{basedir}/sd15/controlnet/openpose",
            ControlNetModelKeys.Canny_sdxl: f"{basedir}/sdxl/controlnet/canny",
            ControlNetModelKeys.Depth_sdxl: f"{basedir}/sdxl/controlnet/depth",
            ControlNetModelKeys.OpenPose_sdxl: f"{basedir}/sdxl/controlnet/controlnet-dwpose-sdxl-1.0",
            IPAdapterEncoderKeys.sd_v15: f"{basedir}/sd15/ip-adapter/ip_adapter_sd_image_encoder",
            IPAdapterEncoderKeys.sdxl: f"{basedir}/sdxl/ip-adapter/ip_adapter_sdxl_image_encoder",
            IPAdapterKeys.sd_v15: f"{basedir}/sd15/ip-adapter/ip_adapter_sd15/ip_adapter.bin",
            IPAdapterKeys.sdxl: f"{basedir}/sdxl/ip-adapter/ip_adapter_sdxl/ip_adapter.bin",
        }

    def reset_paths(self, to_default: bool = False):
        """reset all base model paths to empty or default"""
        self.base_model_paths = {}
        if to_default:
            self._init_default_paths()

    def set_use_shared_model_components(self, use_shared: bool):
        """Different stable diffusion models usually use the same vae, text encoder, etc.
        We may avoid loading the same components twice by allowing the system to use shared model
        components. If this is set to True, shared model components will have the same key
        across different diffusion models, thus they are considered the same.
        """
        self.use_shared_model_components = use_shared

    def set_base_model_path(self, model_key: str, path: str):
        """set the base model path

        parameters
        ------------
        model_key : str
            key of the model, in SDModelKeys or SDXLModelKeys
        path : str
            path to the model folder. If None, remove the path.
        """
        import os

        all_model_keys = (
            SDModelKeys.get_enum_values()
            + SDXLModelKeys.get_enum_values()
            + ControlNetModelKeys.get_enum_values()
            + IPAdapterEncoderKeys.get_enum_values()
            + IPAdapterKeys.get_enum_values()
        )

        is_model_key_valid = model_key in all_model_keys
        assert is_model_key_valid, f"unknown model key {model_key}"

        if path is None:
            self.base_model_paths.pop(model_key, None)
        else:
            assert os.path.exists(path), f"path {path} does not exist"
            self.base_model_paths[model_key] = path

    def get_base_model_paths(self) -> dict[str, str]:
        """get the model folders"""
        return self.base_model_paths

    def get_all_model_paths(self) -> dict[str, str]:
        """return a dict of model paths, key is the model key, value is the path.
        This includes all individual components of the models.
        """
        basepaths = self.base_model_paths
        all_paths = basepaths.copy()

        # add all models into path
        for sdkey in SDModelKeys.get_enum_values() + SDXLModelKeys.get_enum_values():
            if sdkey not in all_paths:
                continue

            mdconfig: SDModelConfig = self.get_model_config(sdkey)
            for ff in fields(mdconfig.__class__):
                fname = ff.name
                fval = getattr(mdconfig, fname)
                if fval is None:
                    continue

                if isinstance(fval, TextModelComponent):
                    if (
                        fval.encoder.path is not None
                        and fval.encoder.key not in all_paths
                    ):
                        assert os.path.exists(
                            fval.encoder.path
                        ), f"path {fval.encoder.path} does not exist"
                        all_paths[fval.encoder.key] = fval.encoder.path
                    if (
                        fval.tokenizer.path is not None
                        and fval.tokenizer.key not in all_paths
                    ):
                        assert os.path.exists(
                            fval.tokenizer.path
                        ), f"path {fval.tokenizer.path} does not exist"
                        all_paths[fval.tokenizer.key] = fval.tokenizer.path

                elif (
                    isinstance(fval, ModelComponent)
                    and fval.path is not None
                    and fval.key not in all_paths
                ):
                    assert os.path.exists(fval.path), f"path {fval.path} does not exist"
                    all_paths[fval.key] = fval.path

        return all_paths

    def get_model_config(self, model_key: str) -> SDModelConfig:
        """given the model key, get the model config with all paths filled in."""
        mdconfig: SDModelConfig = None

        if model_key in SDModelKeys.get_enum_values():  # this is 1.5
            if self.use_shared_model_components:
                SharedModel = (
                    SDModelKeys.get_base_model()
                )  # we expect all sd models share some components with base model
            else:
                SharedModel = model_key

            mdconfig = SDModelConfig(
                primary_model_key=model_key,
                text_model=TextModelComponent(
                    key=SharedModel + "." + TextModelKeys.CLIP_TextModel_prompt_1,
                    encoder=ModelComponent(
                        key=SharedModel + "." + TextModelKeys.CLIP_TextEncoder_prompt_1
                    ),
                    tokenizer=ModelComponent(
                        key=SharedModel
                        + "."
                        + TextModelKeys.CLIP_TextTokenizer_prompt_1
                    ),
                ),
                unet=ModelComponent(key=model_key + ".unet"),
                vae=ModelComponent(key=SharedModel + ".vae"),
                scheduler=SchedulerComponent(
                    key=model_key + ".scheduler",
                    scheduler_type=self.DefaultSchedulerType_sd_v15,
                ),
                image_processor=ModelComponent(
                    key=SharedModel + "." + ImageProcessorKeys.CLIP_ImageProcessor
                ),
                controlnets=ControlnetModelConfig(
                    canny=ModelComponent(key=ControlNetModelKeys.Canny_sd_v15),
                    depth=ModelComponent(key=ControlNetModelKeys.Depth_sd_v15),
                    openpose=ModelComponent(key=ControlNetModelKeys.OpenPose_sd_v15),
                ),
                ip_adapter=IPAdapterModelConfig(
                    encoder=ModelComponent(key=IPAdapterEncoderKeys.sd_v15),
                    adapters={
                        IPAdapterKeys.sd_v15: ModelComponent(key=IPAdapterKeys.sd_v15)
                    },
                ),
            )
        elif model_key in SDXLModelKeys.get_enum_values():  # this is SDXL
            if self.use_shared_model_components:
                SharedModel = SDXLModelKeys.get_base_model()
            else:
                SharedModel = model_key

            mdconfig = SDModelConfig(
                primary_model_key=model_key,
                text_model=TextModelComponent(
                    key=SharedModel + "." + TextModelKeys.CLIP_TextModel_prompt_1,
                    encoder=ModelComponent(
                        key=SharedModel + "." + TextModelKeys.CLIP_TextEncoder_prompt_1
                    ),
                    tokenizer=ModelComponent(
                        key=SharedModel
                        + "."
                        + TextModelKeys.CLIP_TextTokenizer_prompt_1
                    ),
                ),
                text_model_2=TextModelComponent(
                    key=SharedModel + "." + TextModelKeys.CLIP_TextModel_prompt_2,
                    encoder=ModelComponent(
                        key=SharedModel + "." + TextModelKeys.CLIP_TextEncoder_prompt_2
                    ),
                    tokenizer=ModelComponent(
                        key=SharedModel
                        + "."
                        + TextModelKeys.CLIP_TextTokenizer_prompt_2
                    ),
                ),
                unet=ModelComponent(key=model_key + ".unet"),
                vae=ModelComponent(key=SharedModel + ".vae"),
                scheduler=SchedulerComponent(
                    key=model_key + ".scheduler",
                    scheduler_type=self.DefaultSchedulerType_sdxl,
                ),
                controlnets=ControlnetModelConfig(
                    canny=ModelComponent(key=ControlNetModelKeys.Canny_sdxl),
                    depth=ModelComponent(key=ControlNetModelKeys.Depth_sdxl),
                    openpose=ModelComponent(key=ControlNetModelKeys.OpenPose_sdxl),
                ),
                ip_adapter=IPAdapterModelConfig(
                    encoder=ModelComponent(key=IPAdapterEncoderKeys.sdxl),
                    adapters={
                        IPAdapterKeys.sdxl: ModelComponent(key=IPAdapterKeys.sdxl)
                    },
                ),
            )
        else:
            assert False, f"unknown model key {model_key}"

        # assign paths
        p_base = self.get_base_model_paths().get(model_key)
        if p_base is None:
            return mdconfig

        mdconfig.vae.path = f"{p_base}/vae"
        mdconfig.unet.path = f"{p_base}/unet"
        mdconfig.text_model.encoder.path = f"{p_base}/text_encoder"
        mdconfig.text_model.tokenizer.path = f"{p_base}/tokenizer"
        mdconfig.scheduler.path = f"{p_base}/scheduler/scheduler_config.json"

        if mdconfig.image_processor is not None:
            mdconfig.image_processor.path = f"{p_base}/feature_extractor"

        if mdconfig.text_model_2 is not None:
            mdconfig.text_model_2.encoder.path = f"{p_base}/text_encoder_2"
            mdconfig.text_model_2.tokenizer.path = f"{p_base}/tokenizer_2"

        if mdconfig.controlnets is not None:
            if mdconfig.controlnets.canny is not None:
                path = self.get_base_model_paths().get(mdconfig.controlnets.canny.key)
                if os.path.isdir(path):
                    mdconfig.controlnets.canny.path = path
            if mdconfig.controlnets.depth is not None:
                path = self.get_base_model_paths().get(mdconfig.controlnets.depth.key)
                if os.path.isdir(path):
                    mdconfig.controlnets.depth.path = path
            if mdconfig.controlnets.openpose is not None:
                path = self.get_base_model_paths().get(
                    mdconfig.controlnets.openpose.key
                )
                if os.path.isdir(path):
                    mdconfig.controlnets.openpose.path = path

        if mdconfig.ip_adapter is not None:
            if mdconfig.ip_adapter.encoder is not None:
                path = self.get_base_model_paths().get(mdconfig.ip_adapter.encoder.key)
                if os.path.isdir(path):
                    mdconfig.ip_adapter.encoder.path = path

            if mdconfig.ip_adapter.adapters is not None:
                for k, v in mdconfig.ip_adapter.adapters.items():
                    path = self.get_base_model_paths().get(v.key)
                    if os.path.isfile(path):
                        v.path = path

        return mdconfig

    @classmethod
    def get_default_model_root(cls) -> str:
        # get absolute path of 'models' dir, relative to this file
        return str(
            pathlib.Path(__file__).parent.parent.absolute() / "data/models"
        ).replace("\\", "/")

    @classmethod
    def get_default_bento_model_dir(cls) -> str:
        return str(
            pathlib.Path(__file__).parent.parent.absolute() / "data/bento-models"
        ).replace("\\", "/")

    @classmethod
    def setup_bento_model_dir(cls) -> str:
        """set up the environment for bento models."""
        import os

        os.environ["BENTOML_HOME"] = cls.get_default_bento_model_dir()


class Logme(CustomLogger):
    _instance: "Logme" = None
    _default_logger_name = "pyaigc"

    def __init__(self) -> None:
        super().__init__(
            logger_name=self._default_logger_name,
            handler_name=self._default_logger_name,
        )

    @classmethod
    def get_instance(cls) -> "Logme":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
