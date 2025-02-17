import numpy as np
import torch
from attrs import define, field

from .StableDiffusionWithControl import (
    DiffusionState,
    ControlNetInfo,
    ControlnetData,
    TextPromptData,
    IDiffusionTextModel, IDiffusionUnet, 
    IDiffusionVae, IControlnet, DiffusionScheduler,
    StableDiffusionWithControl
)

from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from igpy.common.shortfunc import to_4d_tensor

@define(kw_only=True)
class SDXLDiffusionState(DiffusionState):
    positive_prompt_2 : TextPromptData | None = field(default=None)
    negative_prompt_2 : TextPromptData | None = field(default=None)

class SDXLWithControl(StableDiffusionWithControl):
    def __init__(self) -> None:
        super().__init__()
        self.m_text_model_2 : IDiffusionTextModel = None
    
    # copy from diffusers sdxl pipeline
    def _diffusers_get_add_time_ids(self, original_size : torch.Size, 
                          crops_coords_top_left : torch.Size, 
                          target_size : torch.Size, 
                          text_encoder_projection_dim : int) -> torch.Tensor:
        ''' get the additional time ids for the unet, copy from diffusers
        
        parameters
        ---------------
        original_size : np.ndarray
            original image size, in (height, width)
        crops_coords_top_left : np.ndarray
            top left coordinates of the crops, if None, default to (0,0)
        target_size : np.ndarray
            target image size, in (height, width)
        text_encoder_projection_dim : int
            projection dimension of the text 2
            
        return
        --------
        add_time_ids : torch.Tensor
            additional time ids for the unet
        '''
        add_time_ids = np.concatenate([original_size, crops_coords_top_left, target_size]).astype(int)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        addition_time_embed_dim : int = self.m_unet.config['addition_time_embed_dim']

        passed_add_embed_dim = (
            addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim : int = self.m_unet.config['projection_class_embeddings_input_dim']

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids
    
    def _get_add_time_ids(self, dstate : SDXLDiffusionState) -> torch.Tensor:
        ''' get the additional time ids for the unet
        
        parameters
        ---------------
        dstate : SDXLDiffusionState
            diffusion state
            
        return
        --------
        add_time_ids : torch.Tensor
            additional time ids for the unet
        '''
        original_size = torch.Size([dstate.target_height, dstate.target_width])
        crops_coords_top_left = torch.Size([0,0])
        target_size = original_size
        text_encoder_projection_dim = dstate.positive_prompt_2.projected_embedding.shape[-1]
        
        return self._diffusers_get_add_time_ids(original_size=original_size, 
                                                crops_coords_top_left=crops_coords_top_left, 
                                                target_size=target_size, 
                                                text_encoder_projection_dim=text_encoder_projection_dim)
    
    def init_diffusion_state(self, dstate : SDXLDiffusionState, 
                        reset_latent : bool = True,
                        reset_timestep : bool = True,
                        target_device : torch.device = None,
                        target_dtype : torch.dtype = None,
                        batch_size : int = 1) -> None:
        
        if target_device is None:
            target_device = self.m_unet.device
            
        # encode text prompt, SDXL differs from SD1.5 by having prompt 2
        if dstate.positive_prompt_2 is None:
            dstate.positive_prompt_2 = TextPromptData(text=dstate.positive_prompt.text)
            
        if dstate.negative_prompt_2 is None:
            dstate.negative_prompt_2 = TextPromptData(text=dstate.negative_prompt.text)
            
        for prompt in [dstate.positive_prompt, dstate.negative_prompt]:
            if prompt.embedding is None:
                res = self.m_text_model.encode(prompt.text, embedding_layer_index=-3)
                prompt.embedding = res.word_embedding.to(device=target_device)
                prompt.projected_embedding = res.projected_embedding.to(device=target_device)
                
        for prompt in [dstate.positive_prompt_2, dstate.negative_prompt_2]:
            if prompt.embedding is None:
                res = self.m_text_model_2.encode(prompt.text, embedding_layer_index=-3)
                prompt.embedding = res.word_embedding.to(device=target_device)
                prompt.projected_embedding = res.projected_embedding.to(device=target_device)
        
        # init other things
        super().init_diffusion_state(dstate, 
                                     reset_latent=reset_latent, 
                                     reset_timestep=reset_timestep, 
                                     target_device=target_device,
                                     target_dtype=target_dtype,
                                     batch_size=batch_size)

    def _step_predict_noise(self, dstate_batch : list[DiffusionState]) -> torch.Tensor:
        ''' perform denoising diffusion for one step, all given dstates must have the same target size and controlnet data,
        and the batch size of any latent must be 1.
        
        parameters
        -----------
        dstate_batch : list[DiffusionState]
            list of diffusion states to be denoised, the given dstates will be batched together.
            dstate.latent must have batch_size == 1, and must have been completely denoised (i.e. next_step_index should not be None)
        '''
        
        self._check_dstate_precondition(dstate_batch)
        
        # only process those diffusion state that has not completed
        next_timestep : list[int] = [dstate.scheduler.next_timestep for dstate in dstate_batch]

        # prepare image latents
        latent = torch.cat([dstate.latent for dstate in dstate_batch]).requires_grad_(False)
        latent_scaled = latent.clone()
        neg_prompt_embeds : list[torch.Tensor] = []
        neg_add_text_embeds : list[torch.Tensor] = []
        
        pos_prompt_embeds : list[torch.Tensor] = []
        pos_add_text_embeds : list[torch.Tensor] = []
        
        for i, dstate in enumerate(dstate_batch):
            sch = dstate.scheduler.scheduler
            t = next_timestep[i]
            latent_scaled[i] = sch.scale_model_input(latent[i], t)
            
            # process positive prompt
            _word_embeds = torch.cat([dstate.positive_prompt.embedding, dstate.positive_prompt_2.embedding], dim=-1)
            # _projected_embeds = torch.cat([dstate.positive_prompt.projected_embedding, dstate.positive_prompt_2.projected_embedding], dim=-1)
            _projected_embeds = dstate.positive_prompt_2.projected_embedding
            pos_prompt_embeds.append(_word_embeds)
            pos_add_text_embeds.append(_projected_embeds)
            
            # process negative prompt
            _word_embeds = torch.cat([dstate.negative_prompt.embedding, dstate.negative_prompt_2.embedding], dim=-1)
            # _projected_embeds = torch.cat([dstate.negative_prompt.projected_embedding, dstate.negative_prompt_2.projected_embedding], dim=-1)
            _projected_embeds = dstate.negative_prompt_2.projected_embedding
            neg_prompt_embeds.append(_word_embeds)
            neg_add_text_embeds.append(_projected_embeds)
            
        neg_pos_latent : torch.Tensor = torch.cat([latent_scaled]*2, dim=0)    # for negative and positive prompt
        neg_pos_prompt_embeds : torch.Tensor = torch.cat(neg_prompt_embeds + pos_prompt_embeds, dim=0)
        neg_pos_add_text_embeds : torch.Tensor = torch.cat(neg_add_text_embeds + pos_add_text_embeds, dim=0)
        neg_pos_add_time_ids : torch.Tensor = self._get_add_time_ids(dstate_batch[0]).repeat(len(dstate_batch)*2, 1)
        
        # this is used in both controlnet and unet
        added_cond_kwargs = {
            'text_embeds' : neg_pos_add_text_embeds,
            'time_ids' : neg_pos_add_time_ids
        }
        
        # run controlnet
        down_block_res_samples : torch.FloatTensor = None
        mid_block_res_sample : torch.FloatTensor = None
        for key, cnet_info in self.m_controlnets.items():
            cnet_model, cent_weight = cnet_info.model, cnet_info.weight
            
            # batch all controlnet input of this type
            cnet_input_cond = []
            condition_data_layout = 'nchw'
            cnet_data_complete = True   # all dstate have this controlnet data ?
            for dstate in dstate_batch:
                cnet_data = dstate.controlnet_data.get(key)
                if cnet_data is None:
                    cnet_data_complete = False
                    break
                d = dstate.controlnet_data[key].image_data
                layout = dstate.controlnet_data[key].image_data_layout
                dnew = to_4d_tensor(d, input_layout=layout, output_layout=condition_data_layout)
                cnet_input_cond.extend([dnew, dnew])    # for negative and positive prompt
    
            if not cnet_data_complete:
                # missing some controlnet data, skip this controlnet
                continue
            
            cnet_input_cond : torch.Tensor = torch.cat(cnet_input_cond)
            cnet_input_weight : torch.Tensor = torch.tensor([dstate.controlnet_data[key].weight for dstate in dstate_batch]*2)
            t = np.concatenate([next_timestep]*2)
            
            cnet_weight_final = cnet_input_weight * cent_weight
            if torch.abs(cnet_weight_final).max() < self.ControlnetMinimumWeight:
                # skip this controlnet because the weight is too small
                continue
            
            # compute controlnet output
            res = cnet_model.forward(
                sample=neg_pos_latent,
                timestep=t,
                prompt_embedding=neg_pos_prompt_embeds,
                condition=cnet_input_cond,
                condition_data_layout=condition_data_layout,
                conditioning_scale=cnet_input_weight * cent_weight,
                added_cond_kwargs=added_cond_kwargs
            )
            
            if down_block_res_samples is None:
                down_block_res_samples = res.down_block_res_samples
            else:
                down_block_res_samples += res.down_block_res_samples
                
            if mid_block_res_sample is None:
                mid_block_res_sample = res.mid_block_res_sample
            else:
                mid_block_res_sample += res.mid_block_res_sample
                
        # collect ip adapter embeds
        # len(ipa_embeds) = number of ip adapters
        # ipa_embeds[i].shape = (2N, 1, D), N=batch size, D=embedding dimension
        # ipa_embeds[i][0:N] = negative image prompt, ipa_embeds[i][N:2N] = positive image prompt
        if dstate_batch[0].image_prompt_data:
            ipa_embeds = []
            for key in sorted(dstate_batch[0].image_prompt_data.keys()):
                ipa_positive = []
                ipa_negative = []
                for dstate in dstate_batch:
                    ipa_positive.append(dstate.image_prompt_data[key].positive_embedding.reshape((1,-1)))
                    ipa_negative.append(dstate.image_prompt_data[key].negative_embedding.reshape((1,-1)))
                _embeds = torch.stack(ipa_negative + ipa_positive)
                ipa_embeds.append(_embeds)
            
            # ipa_embeds = torch.stack(_ipa_embeds).to(device=latent.device, dtype=latent.dtype)
            added_cond_kwargs['image_embeds']= ipa_embeds
            
        # run unet to get noise component
        t = np.concatenate([next_timestep]*2)
        noise_pred : torch.Tensor = self.m_unet.forward(
            sample=neg_pos_latent,
            timestep=t,
            prompt_embedding=neg_pos_prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # denoise
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * dstate.guidance_scale
        
        return noise_pred