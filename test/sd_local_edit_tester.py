import torch
import numpy as np
from numpy.typing import NDArray

from attrs import define, field, asdict
import attrs.validators as av
from rich import print as pprint
import igpy.common.image_processing as ip
import igpy.common.inout as cio

from test.sd_tester import SDTester, InputCase, OutputCase
import pyaigc.GlobalConfig as C

@define(kw_only=True, eq=False)
class LocalEditInputCase(InputCase):
    fn_inner_mask : str | None = field(default=None)
    inner_mask : np.ndarray | None = field(default=None)
    
    # bounding box that encloses the inner_mask
    # None means full image
    outer_bbox_xywh : np.ndarray | None = field(default=None)
    
    # scale up the outer bbox to this length, then do the inpainting
    outer_bbox_scaled_long_edge : int | None = field(default=None)
    
    @outer_bbox_scaled_long_edge.validator
    def _validate_outer_bbox_scaled_long_edge(self, attr, value):
        if value is not None:
            assert value % 8 == 0, f'outer_bbox_scaled_long_edge must be divisible by 8, got {value}'
            
    def crop_by(self, xmin: int, ymin: int, w: int, h: int):
        super().crop_by(xmin, ymin, w, h)
        self.inner_mask = self.inner_mask[ymin:ymin+h, xmin:xmin+w]
        if self.outer_bbox_xywh is not None:
            self.outer_bbox_xywh[0] -= xmin
            self.outer_bbox_xywh[1] -= ymin
            
            # clip the box
            self.outer_bbox_xywh[0] = max(self.outer_bbox_xywh[0], 0)
            self.outer_bbox_xywh[1] = max(self.outer_bbox_xywh[1], 0)
            self.outer_bbox_xywh[2] = min(self.outer_bbox_xywh[2], w)
            self.outer_bbox_xywh[3] = min(self.outer_bbox_xywh[3], h)
            
    def resize_to(self, h: int, w: int, divisible_by : int = None):
        h_org, w_org = self.height, self.width
        
        super().resize_to(h, w, divisible_by=divisible_by)
        h_new, w_new = self.height, self.width
        
        # resize inner mask if exists
        if self.inner_mask is not None:
            self.inner_mask = ip.imresize(self.inner_mask, (h_new,w_new), interp_method='nearest')
        
        # resize outer box
        x_scale = w_new / w_org
        y_scale = h_new / h_org
        if self.outer_bbox_xywh is not None and self.image is not None:
            bbox_new = self.outer_bbox_xywh * np.array([x_scale, y_scale, x_scale, y_scale])
            bbox_new = bbox_new.round().astype(int)
            self.outer_bbox_xywh = bbox_new
    
@define(kw_only=True, eq=False)
class MaskedDenoiseOutputCase(OutputCase):
    # full size input case, without cropping
    input_case_fullsize : LocalEditInputCase = field(default=None)
    
    @property
    def denoise_mask(self) -> np.ndarray:
        ''' alpha mask of the denoise region, uint8 range
        '''
        ic : LocalEditInputCase = self.input_case
        if ic is not None:
            assert isinstance(ic, LocalEditInputCase), f'input_case must be LocalEditInputCase, got {type(ic)}'
            return ic.inner_mask
        
@define(kw_only=True, eq=False)
class ImageBlender:
    # in source size
    fullsize_source : NDArray[np.uint8] | None = field(default=None)
    fullsize_inner_mask : NDArray[np.uint8] | None = field(default=None)
    
    # in source image space, the region to be blended
    region_bbox_xywh : NDArray[np.int_] | None = field(default=None)  # where is the region in fullsize space?
    region_source : NDArray[np.uint8] | None = field(default=None) 
    region_source_mask : NDArray[np.uint8] | None = field(default=None)
    region_denoised_nhwc : NDArray[np.uint8] | None = field(default=None)
    
    # the original output denoised image
    original_denoised : NDArray[np.uint8] | None = field(default=None)
    
    def get_fullsize_image_by_poisson_blending(self) -> NDArray[np.uint8]:
        ''' get the fullsize image by poisson blending
        
        return
        --------
        output : NDArray[np.uint8]
            the fullsize blended image in nhwc format
        '''
        import cv2
        batch_size, region_h, region_w = self.region_denoised_nhwc.shape[:3]
        denoised_center_at_fullsize = np.array([region_w, region_h])//2 + self.region_bbox_xywh[:2]
        blend_images : list[np.ndarray] = []
        for i in range(batch_size):
            img : np.ndarray = cv2.seamlessClone(
                self.region_denoised_nhwc[i], self.fullsize_source, 
                None, denoised_center_at_fullsize, cv2.NORMAL_CLONE)
            blend_images.append(img)
        output = np.stack(blend_images, axis=0)
        return output

@define(kw_only=True, eq=False)
class LocalEditTester:
    sd_tester : SDTester | None = field(default=None)
    
    def init_input_case(self, ic : LocalEditInputCase):
        self.sd_tester.init_input_case(ic)
        
        if ic.inner_mask is None and ic.fn_image is not None:
            mask = cio.imread(ic.fn_inner_mask, is_binary_mask=True)
            mask = ip.imresize(mask, ic.image.shape[:2], interp_method='nearest')
            mask = mask.astype(np.uint8) * 255
            ic.inner_mask = mask
        
    def crop_and_resize_input_case(self, ic: LocalEditInputCase)->LocalEditInputCase:
        ''' crop and resize input case based on the setting inside the input case.
        Note that the given bounding box will be adjusted to divisible-by-8 size by scaling.
        '''
        import copy
        outer_bbox = ic.outer_bbox_xywh
        if outer_bbox is None:
            outer_bbox = np.array([0,0,ic.image.shape[1], ic.image.shape[0]])
        
        # crop the input case and scale it
        ic_crop_scale = LocalEditInputCase(**asdict(copy.deepcopy(ic)))
        if ic.outer_bbox_xywh is not None:
            x,y,w,h = ic.outer_bbox_xywh
            ic_crop_scale.crop_by(x,y,w,h)
            
        # scale it
        h,w = ic_crop_scale.image.shape[:2]
        if ic.outer_bbox_scaled_long_edge is not None:
            scale_factor = ic.outer_bbox_scaled_long_edge / max(w,h)
            w_scaled, h_scaled = np.round(np.array([w,h]) * scale_factor).astype(int)
            w_scaled = w_scaled // 8 * 8
            h_scaled = h_scaled // 8 * 8
        else:
            w_scaled = w // 8 * 8
            if w_scaled < w:
                w_scaled += 8
                
            h_scaled = h // 8 * 8
            if h_scaled < h:
                h_scaled += 8
            
        assert w_scaled > 0 and h_scaled > 0, f'bbox region is too small after scaling and divisible-by-8 clipping, w_scaled={w_scaled}, h_scaled={h_scaled}'
        ic_crop_scale.resize_to(h_scaled, w_scaled)  
            
        return ic_crop_scale
    
    def make_masked_output_case(self, ic: LocalEditInputCase) -> MaskedDenoiseOutputCase:
        ''' make a denoise output case based on the input case
        
        parameters
        -----------
        ic : LocalEditInputCase
            input case
        
        return
        ---------
        oc : MaskedDenoiseOutputCase
            output case
        '''
        
        # crop selected region and init the local diffusion state
        ic_select_region = self.crop_and_resize_input_case(ic)
        _oc = self.sd_tester.init_output_case(ic_select_region)
        oc_select_region = MaskedDenoiseOutputCase(**asdict(_oc, recurse=False))
        oc_select_region.input_case = ic_select_region
        oc_select_region.input_case_fullsize = ic
        
        # setup denoise mask
        dstate = oc_select_region.dstate
        inner_mask = ic_select_region.inner_mask
        scaled_mask = ip.imresize(inner_mask, dstate.latent.shape[2:], interp_method='bilinear')
        if scaled_mask.dtype == np.uint8:
            scaled_mask = scaled_mask.astype(np.float32) / 255.0
        elif scaled_mask.dtype == bool:
            scaled_mask = scaled_mask.astype(np.float32)
        else:
            scaled_mask = scaled_mask.astype(np.float32)
        
        # resize the mask to latent size
        denoise_mask : torch.Tensor = torch.tensor(scaled_mask)
        denoise_mask = denoise_mask[None,None,...].expand_as(dstate.latent).to(device=dstate.latent.device, dtype=dstate.latent.dtype)
        dstate.latent_denoise_mask = denoise_mask
        
        # encode image into latent
        cur_latent = self.sd_tester.model_pipeline.encode_image_to_latent(
            torch.tensor(ic_select_region.image[None,...]),
            input_data_layout='nhwc')
        cur_latent = cur_latent.to(device=dstate.latent.device, dtype=dstate.latent.dtype).expand_as(dstate.latent)
        dstate.latent = cur_latent.detach().clone()
        # dstate.latent_clean = cur_latent.detach().clone()
        
        return oc_select_region
    
    def get_fullsize_denoised_image(self, oc : MaskedDenoiseOutputCase) -> np.ndarray:
        ''' get the denoised image from the output case, in the original image size.
        
        parameters
        -----------
        oc : MaskedDenoiseOutputCase
            the output case whose latent is fully denoised
            
        return
        -------
        img_nhwc : np.ndarray
            4d the denoised image in the original size, in (batch, h, w, c) format
        '''
        ic_full = oc.input_case_fullsize
        img_batch = self.sd_tester.model_pipeline.decode_latent_to_image(
            oc.dstate.latent,force_upcast=True,output_dtype=torch.uint8)
        fullsize_imgs = []
        for img_region in img_batch:
            img_combine = np.zeros_like(ic_full.image)
            
            # resize the region to match outer bbox of the fullsize image
            x_outer, y_outer, w_outer, h_outer = ic_full.outer_bbox_xywh
            img_region_resized = ip.imresize(img_region, (h_outer, w_outer), interp_method='bilinear')
            
            # paste the region there
            img_combine[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer, ...] = img_region_resized
            fullsize_imgs.append(img_combine)
            
        img_nhwc = np.stack(fullsize_imgs, axis=0)
            
        return img_nhwc
    
    def init_image_blending(self, oc : MaskedDenoiseOutputCase)-> ImageBlender:
        ''' prepare the different parts for image blending
        
        parameters
        -----------
        oc : MaskedDenoiseOutputCase
            the output case whose latent is fully denoised
        
        return
        -------
        output : ImageBlender
            the image blender object containing all decoded images
        '''
        ic_full = oc.input_case_fullsize
        x,y,w,h = ic_full.outer_bbox_xywh
        region_source = ic_full.image[y:y+h, x:x+w, ...]
        if ic_full.inner_mask is not None:
            region_source_mask = ic_full.inner_mask[y:y+h, x:x+w, ...]
        else:
            region_source_mask = None
        
        # denoise image
        img_denoised = self.sd_tester.model_pipeline.decode_latent_to_image(
            oc.dstate.latent, force_upcast=True, output_dtype=torch.uint8
        )
        batch_size = len(img_denoised)
        
        # resize denoised image to region
        _, _, w_outer, h_outer = ic_full.outer_bbox_xywh
        region_denoised = np.stack([region_source] * batch_size, axis=0)
        for i in range(batch_size):
            region_denoised[i] = ip.imresize(img_denoised[i], (h_outer, w_outer), interp_method='bilinear')
        
        bi = ImageBlender(
            fullsize_source=ic_full.image,
            fullsize_inner_mask=ic_full.inner_mask,
            region_bbox_xywh=ic_full.outer_bbox_xywh,
            region_source=region_source,
            region_source_mask=region_source_mask,
            region_denoised_nhwc=region_denoised,
            original_denoised=img_denoised
        )
        
        return bi
