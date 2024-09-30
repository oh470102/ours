from dataclasses import dataclass
import os
import pickle
import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable, List, Optional, Union, Any, Dict
from diffusers.utils import deprecate, logging, BaseOutput, randn_tensor
from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
import kornia
import gc

#
import torchvision.transforms as transforms

pad_mode = 'center'

@torch.no_grad()
def customized_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,

        # Guidance parameters
        score=None,
        guidance_scale=0.0,
        indices=None, 

):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    
    # Support IF models
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)

    # ? optional 
    #########################
    # eta = 0.0
    eta = 0.025
    #########################
    # ?
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5) # [2, 4, 64, 64]

    # 6. apply guidance following the formula (14) from https://arxiv.org/pdf/2105.05233.pdf
    if score is not None and guidance_scale > 0.0: 

        # ? optional 
        #########################
        # score = std_dev_t * score 
        # score  = score / (torch.std(score) + 1e-7)
        score = std_dev_t * score / (torch.std(score) + 1e-7)
        #########################
        # ? 

        if indices is not None:
            # import pdb; pdb.set_trace()
            assert pred_epsilon[indices].shape == score.shape, "pred_epsilon[indices].shape != score.shape"
            pred_epsilon[indices] = pred_epsilon[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score 
        else:
            assert pred_epsilon.shape == score.shape
            pred_epsilon = pred_epsilon - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score

    # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon 

    # 8. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction 

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise 

        prev_sample = prev_sample + variance 
    self.pred_epsilon = pred_epsilon
    if not return_dict:
        return (prev_sample,)

    return prev_sample, pred_original_sample

def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,timestep_spacing_type= "linspace"):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps

    # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
    if timestep_spacing_type == "linspace":
        timesteps = (
            np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
    elif timestep_spacing_type == "leading":
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps += self.config.steps_offset
    elif timestep_spacing_type == "trailing":
        step_ratio = self.config.num_train_timesteps / self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
        timesteps -= 1
    else:
        raise ValueError(
            f"{timestep_spacing_type} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
        )

    self.timesteps = torch.from_numpy(timesteps).to(device)

def find_centroid(mask):
    B, H, W = mask.shape
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device).view(1, 1, W).expand(B, H, W)
    
    total_mass = mask.sum(dim=(1, 2))
    y_center = (y_coords * mask).sum(dim=(1, 2)) / total_mass
    x_center = (x_coords * mask).sum(dim=(1, 2)) / total_mass
    
    return torch.stack([y_center, x_center], dim=1)

def compute_centroid(diff_map):
    """Compute the centroid of non-zero regions in the diff_map."""

    diff_map = diff_map.float()  # Convert to FP32 to ensure stability

    h_indices = torch.arange(diff_map.shape[0], device=diff_map.device)
    w_indices = torch.arange(diff_map.shape[1], device=diff_map.device)

    H, W = torch.meshgrid(h_indices, w_indices, indexing="ij")

    total_mass = diff_map.sum()

    if total_mass == 0:
        return torch.tensor([0.0, 0.0], device=diff_map.device)  # No significant change

    centroid_h = (H * diff_map).sum() / (total_mass + 1e-5)
    centroid_w = (W * diff_map).sum() / (total_mass + 1e-5)

    return torch.stack([centroid_h, centroid_w]).half()


def get_gaussian_blur(mask, sigma=5):
    mask = mask.reshape(mask.shape[0],int(mask.shape[1]**0.5),int(mask.shape[1]**0.5))
    B, H, W = mask.shape
    centroids = find_centroid(mask)
    
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device).view(1, 1, W).expand(B, H, W)
    
    y_centers = centroids[:, 0].view(B, 1, 1).expand(B, H, W)
    x_centers = centroids[:, 1].view(B, 1, 1).expand(B, H, W)
    
    gauss_y = torch.exp(-((y_coords - y_centers) ** 2) / (2 * sigma ** 2))
    gauss_x = torch.exp(-((x_coords - x_centers) ** 2) / (2 * sigma ** 2))
    gauss = gauss_y * gauss_x
    
    return gauss.reshape(mask.shape[0],-1,1)

def compute_cross_attn_mask(mask_blocks, cross_attn2_prob,token_index_example, token_index_app, mask_threshold_example=0.2,mask_threshold_app=0.3,step_index=None, get_all = False):

    mask_example_foreground,mask_example_background,mask_app_foreground,mask_app_background  = {}, {}, {}, {}
    
    for block_name in mask_blocks:
        if block_name != "up_blocks.1":
            # [frame, H*W, 1]
            feature = mask_example_foreground["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            # import pdb; pdb.set_trace()
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_example_foreground[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)  
            
            feature = mask_example_background["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_example_background[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
            
            feature = mask_app_foreground["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_app_foreground[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
            
            feature = mask_app_background["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_app_background[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
                                                        
        else:
            attn2_prob_example = []
            attn2_prob_app = []
            corss_attn2_prob_each_block = {key: cross_attn2_prob[key] for key in cross_attn2_prob.keys() if block_name in key}
            
            for name in corss_attn2_prob_each_block.keys():

                cross_attn2_prob_each = corss_attn2_prob_each_block[name]

                ### * generalize frame number
                
                # double_fh, hw, _= cross_attn2_prob_each.shape         
                # f = double_fh // (2 * 8)                    # 2: one for ref, one for gen,  8: n_heads
                # cross_attn2_prob_each = cross_attn2_prob_each.view(2,f,-1,cross_attn2_prob_each.shape[1],cross_attn2_prob_each.shape[2])
                
                ### *

                cross_attn2_prob_each = cross_attn2_prob_each.reshape(2,16,-1,cross_attn2_prob_each.shape[1],cross_attn2_prob_each.shape[2])
                attn2_prob_example.append(cross_attn2_prob_each[0])
                attn2_prob_app.append(cross_attn2_prob_each[1])
                # [16, head, H*W, 77]
                
        

            attn2_prob_example = torch.mean(torch.cat(attn2_prob_example, dim = 1),dim=1)
            attn2_prob_app = torch.mean(torch.cat(attn2_prob_app, dim = 1),dim=1)
            
            # [frame, H*W, 1]
            mask_example = attn2_prob_example[:,:,[token_index_example]]
            mask_example = (mask_example - mask_example.min(dim=1,keepdim=True)[0])/(mask_example.max(dim=1,keepdim=True)[0]-mask_example.min(dim=1,keepdim=True)[0]+1e-5)

            ######################### 
            # mask_example_foreground[block_name] = (mask_example > mask_threshold_example).to(attn2_prob_example.dtype)
            # mask_example_background[block_name] = 1-mask_example_foreground[block_name]
            #########################
            
            mask_no_blur = (mask_example>mask_threshold_example).to(attn2_prob_example.dtype)
            gaussian_blur = get_gaussian_blur(mask_no_blur,sigma=5).to(mask_no_blur.dtype)
            mask_example_foreground[block_name] = gaussian_blur* mask_no_blur
            mask_example_background[block_name] = (1-gaussian_blur)*(1-mask_no_blur) 
            
            
            mask_app = attn2_prob_app[:,:,[token_index_app]]
            mask_app = (mask_app - mask_app.min(dim=1,keepdim=True)[0])/(mask_app.max(dim=1,keepdim=True)[0]-mask_app.min(dim=1,keepdim=True)[0]+1e-5)

            mask_app_foreground[block_name] = (mask_app>mask_threshold_app).to(attn2_prob_app.dtype)
            mask_app_background[block_name] = 1-mask_app_foreground[block_name]


        if step_index is not None and step_index == 250:
            for index in range(mask_example_foreground[block_name].shape[0]):
                mask_example_each = mask_example_foreground[block_name][index]
                res = int(np.sqrt(mask_example_each.shape[0]))
                mask_example_each = mask_example_each.reshape(res,res).cpu().numpy() * 255
                mask_example_each =  Image.fromarray(mask_example_each.astype(np.uint8))
                save_path = os.path.join("masks","example_"+ block_name +"_" +str(step_index) +"_" +str(index)+".png")
                mask_example_each.save(save_path)
                
                mask_app_each = mask_app_foreground[block_name][index]
                mask_app_each = mask_app_each.reshape(res,res).cpu().numpy() * 255
                mask_app_each =  Image.fromarray(mask_app_each.astype(np.uint8))
                save_path = os.path.join("masks","app_"+ block_name +"_" +str(step_index) +"_" +str(index)+".png")
                mask_app_each.save(save_path)
    
    return mask_example_foreground, mask_example_background, mask_app_foreground, mask_app_background

def compute_temp_loss(temp_attn_prob_example, temp_attn_prob_control, weight_each,mask_example, step_index=None):

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]
        
        # (bhw, n_head, f, f)
        module_attn_loss = calculate_motion_rank(attn_prob_example.detach(), attn_prob_control, rank_k = 1)
        temp_attn_prob_loss.append(module_attn_loss)
            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each
    return loss_temp.mean()

def calculate_motion_rank(tensor_ref, tensor_gen, rank_k = 1, use_soft_mse=False, use_weight=False):
    if rank_k ==0:
        loss = torch.tensor(0.0,device = tensor_ref.device)
    elif rank_k > tensor_ref.shape[-1]:
        raise ValueError("the value of rank_k cannot larger than the number of frames")
    else:   
        # sort tensor in ascending order
        _, sorted_indices = torch.sort(tensor_ref, dim=-1)

        # zeros of shape [64, 8, 16, 16-rank] + ones of shape [64, 8, 16, rank]
        # zeros: values to be masked,  ones: top-k values
        mask_indices = torch.cat((torch.zeros([*tensor_ref.shape[:-1],tensor_ref.shape[-1]-rank_k], dtype=torch.bool), torch.ones([*tensor_ref.shape[:-1],rank_k], dtype=torch.bool)),dim=-1)
        
        # retrieve the maximum values from reference tensor, expand it from [64, 8, 16, 1] --> [64, 8, 16, 16]
        max_copy = sorted_indices[:,:,:,[-1]].expand(*tensor_ref.shape[:-1], tensor_ref.shape[-1])
        sorted_indices[~mask_indices] = max_copy[~mask_indices]
        
        # create actual mask based on indices
        mask = torch.zeros_like(tensor_ref, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, True) 

        # get final MSE loss
        loss = F.mse_loss(tensor_ref[mask].detach(), tensor_gen[mask])
        
    return loss

def get_object_index(tokenizer, prompt: str, word: str,):
    tokens_list = tokenizer(prompt.lower()).input_ids
    search_tokens = tokenizer(word.lower()).input_ids
    token_index = tokens_list.index(search_tokens[1])
    return token_index

def compute_semantic_loss( temp_attn_key_app, temp_attn_key_control, weight_each,mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type="temp"):
    temp_attn_key_loss = []
    for name in temp_attn_key_app.keys():

        attn_key_app = temp_attn_key_app[name]
        attn_key_control = temp_attn_key_control[name]
        if mask_example_fore == None:
            module_attn_loss = calculate_semantic_loss(attn_key_app.detach(), attn_key_control, 
                                                    None, None, None, None, block_type)
        else:
            block_name = ".".join(name.split(".")[:2])
            module_attn_loss = calculate_semantic_loss(attn_key_app.detach(), attn_key_control, 
                                                        mask_example_fore[block_name], mask_example_back[block_name],mask_app_fore[block_name], mask_app_back[block_name],block_type)

        temp_attn_key_loss.append(module_attn_loss)
            
    loss_app = torch.stack(temp_attn_key_loss) * weight_each
    return loss_app.mean()

def calculate_semantic_loss(tensor_app, tensor_gen, mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type):
 
    if mask_example_fore is None:
        if block_type == "temp":
            # loss = F.mse_loss(tensor_app.mean(dim=0).detach(), tensor_gen.mean(dim=0))
            loss = F.mse_loss(tensor_app.mean(dim=[0,1],keepdim=True).detach(), tensor_gen.mean(dim=0,keepdim=True))  
        else:
            # loss = F.mse_loss(tensor_app.mean(dim=1).detach(), tensor_gen.mean(dim=1))  
            loss = F.mse_loss(tensor_app.mean(dim=[0,1], keepdim=True).detach(), tensor_gen.mean(dim=1,keepdim=True))  
 
    else:
        if block_type == "temp":
            tensor_app = tensor_app.permute(1,0,2)
            tensor_gen = tensor_gen.permute(1,0,2)
        # [frame, H*W, head*dim]

        ref_foreground = (tensor_app*mask_app_fore).sum(dim=1)/(mask_app_fore.sum(dim=1)+1e-5)
        ref_background = (tensor_app*mask_app_back).sum(dim=1)/(mask_app_back.sum(dim=1)+1e-5)
        
        gen_foreground = (tensor_gen*mask_example_fore).sum(dim=1)/(mask_example_fore.sum(dim=1)+1e-5)
        gen_background = (tensor_gen*mask_example_back).sum(dim=1)/(mask_example_back.sum(dim=1)+1e-5)
        
        loss =  F.mse_loss(ref_foreground.detach(), gen_foreground) +  F.mse_loss(ref_background.detach(), gen_background) 
        # loss = F.mse_loss(ref_background.detach(), gen_background) 
        # loss =  F.mse_loss(ref_foreground.detach(), gen_foreground)

    return loss

# * BELOW ARE IMPLEMENTED BY ME



# * util

def random_sample_tensor(tensor, sample_percentage):
    
    # extract input tensor shape
    h, w, f, dim = tensor.shape

    # how many channels to sample?
    num_channels = int(dim * sample_percentage)

    # sample and save indices
    selected_indices = torch.randperm(num_channels)[:num_channels]
    tensor = tensor[..., selected_indices]

    return tensor, selected_indices


def apply_shear_transform(tensor, shear_factor=0.15):
    """
    Apply a shear transformation to a 4D tensor with shape (h, w, f, dim) using kornia.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (h, w, f, dim).
    shear_factor (float): The factor by which to shear the tensor along the width axis.

    Returns:
    torch.Tensor: The sheared tensor with the same shape as the input,
                  consistent in data type and device.
    """
    # Get the dimensions of the tensor
    h, w, f, dim = tensor.shape

    # Reshape the tensor for the transformation
    tensor = tensor.permute(2, 3, 0, 1)  # Shape (f, dim, h, w)
    tensor = tensor.view(f*dim, 1, h, w)

    # Create a shear transformation matrix for the whole batch
    shear_matrix = torch.tensor([[1.0, shear_factor, 0.0], [0.0, 1.0, 0.0]], device=tensor.device, dtype=tensor.dtype)
    shear_matrix = shear_matrix.unsqueeze(0).repeat(f, 1, 1)  # Shape (f, 2, 3)
    
    # Expand the shear matrix to cover all channels in the batch
    shear_matrix = shear_matrix.repeat(dim, 1, 1)  # Shape (f * dim, 2, 3)

    # Apply the shear transformation using kornia
    sheared_tensor = kornia.geometry.transform.warp_affine(tensor, shear_matrix, dsize=(h, w), mode='bicubic', align_corners=True)

    # Reshape back to the original tensor shape
    sheared_tensor = sheared_tensor.view(f, dim, h, w)
    sheared_tensor = sheared_tensor.permute(2, 3, 0, 1)  # Shape back to (h, w, f, dim)

    return sheared_tensor

def select_matching_indices(tensor):
    # Get the shapes of the tensor
    n_head, f1, f2, hw = tensor.shape

    # Create a mask that selects only elements where the indices along dim=1 and dim=2 are equal
    mask = torch.arange(f1)[:, None] == torch.arange(f2)[None, :]

    # Apply the mask
    # We broadcast the mask to the shape of the input tensor and use it to filter the tensor
    result = tensor[:, mask, :]

    return result

def perspective_warp_no_interpolation_tensor(src_tensor, H, output_shape, tolerance=0.2):
    B, C, H_src, W_src = src_tensor.shape
    H_dst, W_dst = output_shape

    # Initialize the output tensor with zeros
    dst_tensor = torch.zeros((B, C, H_dst, W_dst), dtype=src_tensor.dtype, device=src_tensor.device)
    
    # Generate a meshgrid of (x, y) coordinates
    y, x = torch.meshgrid(torch.arange(H_src, device=src_tensor.device), torch.arange(W_src, device=src_tensor.device), indexing='ij')
    ones = torch.ones_like(x, dtype=torch.float32, device=src_tensor.device)
    src_pts = torch.stack([x.float(), y.float(), ones], dim=0).reshape(3, -1)  # (3, H_src * W_src)

    # Apply homography matrix H to the source points
    dst_pts = torch.matmul(H, src_pts)  # (B, 3, H_src * W_src)
    dst_pts = dst_pts / dst_pts[:, 2:3, :]  # Normalize by the third coordinate

    # Extract x and y coordinates
    dst_x, dst_y = dst_pts[:, 0, :], dst_pts[:, 1, :]
    
    # Round to the nearest integer coordinates
    int_dst_x = torch.round(dst_x).long().reshape(B, H_src, W_src)
    int_dst_y = torch.round(dst_y).long().reshape(B, H_src, W_src)
    
    # Create masks for valid points (within bounds and within tolerance)
    mask_x = (int_dst_x >= 0) & (int_dst_x < W_dst)
    mask_y = (int_dst_y >= 0) & (int_dst_y < H_dst)
    mask_tolerance_x = (torch.abs(dst_x.reshape(B, H_src, W_src) - int_dst_x.float()) <= tolerance)
    mask_tolerance_y = (torch.abs(dst_y.reshape(B, H_src, W_src) - int_dst_y.float()) <= tolerance)
    mask = mask_x & mask_y & mask_tolerance_x & mask_tolerance_y

    # Get the valid source coordinates based on the mask
    valid_src_x = x.expand(B, -1, -1)[mask]
    valid_src_y = y.expand(B, -1, -1)[mask]

    # Get the valid destination coordinates based on the mask
    valid_dst_x = int_dst_x[mask]
    valid_dst_y = int_dst_y[mask]

    # Generate a batch of indices for the destination tensor
    batch_indices = torch.arange(B, device=src_tensor.device).view(-1, 1, 1).expand_as(int_dst_x)[mask]

    # Assign values from the source tensor to the destination tensor
    dst_tensor[batch_indices, :, valid_dst_y, valid_dst_x] = src_tensor[batch_indices, :, valid_src_y, valid_src_x]

    return dst_tensor

def shift_tensor(tensor, shift=None):
    '''
    expects input shape (hw, dim, f, f)
    shift: tuple, with positive up, left
    '''
    # setup
    hw, n_head, f1, f2 = tensor.shape
    res = h = w = int(hw**0.5)

    # positive: up, left
    if shift is None: shift = (-h//4, 0)
    shift_x, shift_y = shift

    # roll but with no circular shifting
    tensor = tensor.view(res, res, n_head, f1, f2)
    if shift_x != 0:
        if shift_x > 0:
            tensor = torch.cat((tensor[shift_x:, :, :, :, :], torch.zeros_like(tensor[:shift_x, :, :, :, :])), dim=0)
        else:
            shift_x = -shift_x
            tensor = torch.cat((torch.zeros_like(tensor[-shift_x:, :, :, :, :]), tensor[:-shift_x, :, :, :, :]), dim=0)

    if shift_y != 0:
        if shift_y > 0:
            tensor = torch.cat((tensor[:, shift_y:, :, :, :], torch.zeros_like(tensor[:, :shift_y, :, :, :])), dim=1)
        else:
            shift_y = -shift_y
            tensor = torch.cat((torch.zeros_like(tensor[:, -shift_y:, :, :, :]), tensor[:, :-shift_y, :, :, :]), dim=1)

    # Reshape back to original shape
    shifted_tensor = tensor.view(res, res, n_head, f1, f2)
    
    return shifted_tensor
    
def shift_mask(tensor, shift=None):

    # check layout
    f, hw, b = tensor.shape
    res = h = w = int(hw**0.5)

    # setup
    # positive: up, left
    if shift is None: shift = (0, -w//4)
    shift_x, shift_y = shift

    # reshape
    tensor_reshaped = tensor.view(f, res, res, b)
   
    # roll - but prevent circular shifting
    if shift_x != 0:
        if shift_x > 0:
            tensor_reshaped = torch.cat((tensor_reshaped[:, :, shift_x:, :], torch.zeros_like(tensor_reshaped[:, :, :shift_x, :])), dim=2)
        else:
            shift_x = -shift_x
            tensor_reshaped = torch.cat((torch.zeros_like(tensor_reshaped[:, :, -shift_x:, :]), tensor_reshaped[:, :, :-shift_x, :]), dim=2)

    if shift_y != 0:
        if shift_y > 0:
            tensor_reshaped = torch.cat((tensor_reshaped[:, shift_y:, :, :], torch.zeros_like(tensor_reshaped[:, :shift_y, :, :])), dim=1)
        else:
            shift_y = -shift_y
            tensor_reshaped = torch.cat((torch.zeros_like(tensor_reshaped[:, -shift_y:, :, :]), tensor_reshaped[:, :-shift_y, :, :]), dim=1)

    # reshape-back and return
    return tensor_reshaped.view(f, hw, b)

def soft_threshold(T, s=5, dim=None):
    assert dim is not None, "You must specify a dimension to operate on."

    def normalize(T):
        max_T, _ = torch.max(T, dim=dim, keepdim=True)
        min_T, _ = torch.min(T, dim=dim, keepdim=True)
        return (T - min_T) / (max_T - min_T + 1e-8)  # Add epsilon to avoid division by zero

    T_norm = normalize(T)
    temp1 = T_norm - 0.5
    temp2 = s * temp1
    temp3 = torch.sigmoid(temp2)
    result = normalize(temp3)

    return result

def hard_threshold(tensor, threshold=0.15):
    return torch.where(tensor >= threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

def pad_tensor(tensor, target_size, mode='center'):

    # ? check expected input shape: (B, C, H, W)
    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    if mode=='center':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode=='top-left':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = 0
        pad_right = target_W - W
    
    elif mode=='top-right':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = target_W - W
        pad_right = 0

    elif mode=='bottom-left':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'bottom-right':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-left':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'center-right':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-top':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode == 'center-bottom':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)

    return F.pad(tensor, padding, mode='constant', value=0)

def crop_tensor(tensor, target_size, mode='center'):
    # Extract current and target dimensions
    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    # Compute cropping coordinates based on mode
    if mode == 'center':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'top-left':
        crop_top = 0
        crop_bottom = target_H
        crop_left = 0
        crop_right = target_W
    
    elif mode == 'top-right':
        crop_top = 0
        crop_bottom = target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'bottom-left':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = 0
        crop_right = target_W

    elif mode == 'bottom-right':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-left':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = 0
        crop_right = target_W

    elif mode == 'center-right':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-top':
        crop_top = 0
        crop_bottom = target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'center-bottom':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    # Perform the cropping operation
    cropped_tensor = tensor[:, :, crop_top:crop_bottom, crop_left:crop_right]

    return cropped_tensor

def save_heatmaps(heatmaps, filename=None):

    os.makedirs("Heatmaps", exist_ok=True)
    for i in range(heatmaps.shape[0]):
        # Create a new figure
        plt.figure()

        # Plot the heatmap
        plt.imshow(heatmaps[i], cmap='viridis', aspect='auto', vmin=0, vmax=0.15)

        # Add a color bar with a consistent scale
        plt.colorbar()

        # Set title (optional)
        plt.title(f'F: ({i//16 + 1}, {i%16 + 1})')

        # Save the figure
        plt.savefig(os.path.join("Heatmaps", f'{filename}_heatmap_{i+1}.png'))

        # Close the figure to release memory
        plt.close()

def rotate_tensor(tensor, angle):
    """
    Apply a rotation transformation to a 4D tensor with shape (h, w, f, dim) using kornia.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (h, w, f, dim).
    angle (float): The angle by which to rotate the tensor (in degrees).

    Returns:
    torch.Tensor: The rotated tensor with the same shape as the input,
                  consistent in data type and device.
    """
    # Get the dimensions of the tensor
    h, w, f, dim = tensor.shape

    # Reshape the tensor for the transformation
    tensor = tensor.permute(2, 3, 0, 1)  # Shape (f, dim, h, w)
    tensor = tensor.view(f * dim, 1, h, w)

    # Create a rotation matrix for the entire batch
    rotation_matrix = kornia.geometry.transform.get_rotation_matrix2d(
        center=torch.tensor([[w / 2, h / 2]], device=tensor.device, dtype=tensor.dtype).repeat(f * dim, 1),
        angle=torch.tensor([angle], device=tensor.device, dtype=tensor.dtype).repeat(f * dim),
        scale=torch.ones(f * dim, 2, device=tensor.device, dtype=tensor.dtype)  # Shape (f * dim, 2)
    )  # Shape (f * dim, 2, 3)

    # Apply the rotation transformation using kornia
    rotated_tensor = kornia.geometry.transform.warp_affine(tensor, rotation_matrix, dsize=(h, w))

    # Reshape back to the original tensor shape
    rotated_tensor = rotated_tensor.view(f, dim, h, w)
    rotated_tensor = rotated_tensor.permute(2, 3, 0, 1)  # Shape back to (h, w, f, dim)

    return rotated_tensor

# * motion loss - shift ver.
def compute_temp_loss_with_shift(temp_attn_prob_example, temp_attn_prob_control, weight_each, mask_example):

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]

        #############

        hw, n_head, f, f = attn_prob_example.shape
        attn_prob_example = shift_tensor(attn_prob_example).view(hw, n_head, f, f)

        #############

        module_attn_loss = calculate_motion_rank_with_shift(attn_prob_example.detach(), attn_prob_control, rank_k = 1)
        temp_attn_prob_loss.append(module_attn_loss)
            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each

    return loss_temp.mean()

def calculate_motion_rank_with_shift(tensor_ref, tensor_gen, rank_k = 1, use_soft_mse=False, use_weight=False):
    if rank_k ==0:
        loss = torch.tensor(0.0,device = tensor_ref.device)
    elif rank_k > tensor_ref.shape[-1]:
        raise ValueError("the value of rank_k cannot larger than the number of frames")
    else:   
        # sort tensor in ascending order
        _, sorted_indices = torch.sort(tensor_ref, dim=-1)

        # zeros of shape [64, 8, 16, 16-rank] + ones of shape [64, 8, 16, rank]
        # zeros: values to be masked,  ones: top-k values
        mask_indices = torch.cat((torch.zeros([*tensor_ref.shape[:-1],tensor_ref.shape[-1]-rank_k], dtype=torch.bool), torch.ones([*tensor_ref.shape[:-1],rank_k], dtype=torch.bool)),dim=-1)
        
        # retrieve the maximum values from reference tensor, expand it from [64, 8, 16, 1] --> [64, 8, 16, 16]
        max_copy = sorted_indices[:,:,:,[-1]].expand(*tensor_ref.shape[:-1], tensor_ref.shape[-1])
        sorted_indices[~mask_indices] = max_copy[~mask_indices]
        
        # create actual mask based on indices
        mask = torch.zeros_like(tensor_ref, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, True) 

        # NOTE: only for direct shifting
        # exclude padded elements from reference tensor
        # maximum element is 0 == it was padded in the first place
        flat_ref = tensor_ref[mask].detach()
        flat_gen = tensor_gen[mask]
        flat_gen *= flat_ref != 0
        
        loss = F.mse_loss(flat_ref, flat_gen)

        
    return loss



# * motion loss - resize ver.
def compute_temp_loss_with_resize(temp_attn_prob_example, temp_attn_prob_control, weight_each, mask_example, scale = 0.5, step_index=None):

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]

        # extract dim
        hw, n_head, f, f = attn_prob_example.shape
        h = w = int(hw ** 0.5)
        bsz = n_head * f * f

        # reshape --> (B, C, H, W), where B=F*F*n_head
        original_size = (bsz, 1, h, w)
        attn_prob_example = attn_prob_example.detach().view(*original_size)

        # ? remove later
        # save attention heatmap (before padding)
        # if step_index == 250 and h==10:
        #     heatmaps = attn_prob_example.view(f*f, n_head, h, w)[0:15].mean(dim=1, keepdim=False).detach().cpu().numpy()
        #     save_heatmaps(heatmaps, "original")

        # downsample
        # attn_prob_example_ = F.interpolate(attn_prob_example, size=(h//2, w//2), mode='bicubic', align_corners=True)
        attn_prob_example_ = F.max_pool2d(attn_prob_example, kernel_size=2, stride=2)

        # upsample
        # attn_prob_example = F.interpolate(attn_prob_example_, size=(h, w), mode='bicubic', align_corners=True)
        attn_prob_example = pad_tensor(attn_prob_example_, original_size, mode=pad_mode)

        # ? remove later
        # save attention heatmap (after padding)
        # if step_index == 250 and h==10:
        #     heatmaps = attn_prob_example.view(f*f, n_head, h, w)[0:15].mean(dim=1, keepdim=False).detach().cpu().numpy()
        #     save_heatmaps(heatmaps, "after padding")

        # compute loss
        attn_prob_example = attn_prob_example.view(hw, n_head, f, f)
        module_attn_loss = calculate_motion_rank_with_shift(attn_prob_example, attn_prob_control, rank_k = 1)
        temp_attn_prob_loss.append(module_attn_loss)

            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each * 0.5

    return loss_temp.mean()

def compute_temp_loss_with_resize_v2(temp_attn_prob_example, temp_attn_prob_control, weight_each, mask_example, scale = 0.5, step_index=None):

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    names = [_ for _ in temp_attn_prob_example.keys()]

    for i, name in enumerate(names):

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        # extract map
        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]

        # extract dim
        hw, n_head, f, f = attn_prob_example.shape
        h = w = int(hw ** 0.5)
        bsz = n_head * f * f

        if block_num == 0 : 

            # ? use original method at layer 1
            module_attn_loss = calculate_motion_rank(attn_prob_example.detach(), attn_prob_control, rank_k = 1)

            # ? use downsampling at layer 1
            # # reshape --> (B, C, H, W), where B=F*F*n_head
            # original_size = (bsz, 1, h, w)
            # attn_prob_example = attn_prob_example.detach().view(*original_size)

            # # downsample
            # # attn_prob_example_ = F.interpolate(attn_prob_example, size=(h//2, w//2), mode='bicubic', align_corners=True)
            # attn_prob_example_ = F.max_pool2d(attn_prob_example, kernel_size=2, stride=2)

            # # upsample
            # # attn_prob_example = F.interpolate(attn_prob_example_, size=(h, w), mode='bicubic', align_corners=True)
            # attn_prob_example = pad_tensor(attn_prob_example_, original_size, mode=pad_mode)

            # # compute loss
            # attn_prob_example = attn_prob_example.view(hw, n_head, f, f)
            # module_attn_loss = calculate_motion_rank_with_shift(attn_prob_example, attn_prob_control, rank_k = 1)
            
            temp_attn_prob_loss.append(module_attn_loss)
            continue

        # get previous layer
        elif block_num > 0:
            
            # i.e. up_blocks.2.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0
            prev_name = name[:10] + str(int(name[10])-1) + name[11:]
            attn_prob_example_prev = temp_attn_prob_example[prev_name]

            # reshape --> (B, C, H, W), where B=F*F*n_head
            original_size = (bsz, 1, h, w)
            prev_size = (bsz, 1, h//2, w//2)
            attn_prob_example_prev = attn_prob_example_prev.detach().view(*prev_size)

            # ? remove later
            # if step_index == 250 and h==20:
            #     heatmaps = attn_prob_example.view(f*f, n_head, h, w)[0:15].mean(dim=1, keepdim=False).detach().cpu().numpy()
            #     save_heatmaps(heatmaps, "original")
                
            # use padded map from previous layer
            attn_prob_example = pad_tensor(attn_prob_example_prev, original_size, mode=pad_mode)

            # ? remove later
            # if step_index == 250 and h==20:
            #     heatmaps = attn_prob_example.view(f*f, n_head, h, w)[0:15].mean(dim=1, keepdim=False).detach().cpu().numpy()
            #     save_heatmaps(heatmaps, "padded")

            # compute loss
            attn_prob_example = attn_prob_example.view(hw, n_head, f, f)
            module_attn_loss = calculate_motion_rank_with_shift(attn_prob_example, attn_prob_control, rank_k = 2)
            temp_attn_prob_loss.append(module_attn_loss)

            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each * 0.5

    return loss_temp.mean()



# * motion loss - masked ver
def compute_temp_loss_with_mask(temp_attn_prob_example, temp_attn_prob_control, weight_each, 
                                mask_example_fore, step_index=None, use_downsized=False, do_shift=False, 
                                animal_name = None):

    # ?
    assert animal_name is not None, "animal name is None"
    # ?

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        # retrieve mask
        mask_key = f'up_blocks.{block_num}'
        mask_example_input = mask_example_fore[mask_key]
        mask_example_input = rearrange(mask_example_input, 'f hw b -> hw f b') # b=1

        # retrieve attention maps
        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]

        # shift?
        if do_shift:
            hw, n_head, f, f = attn_prob_example.shape
            attn_prob_example = shift_tensor(attn_prob_example).view(hw, n_head, f, f)
        
        # use downsized reference?
        if use_downsized == False: downsized_tensor_ref = None
        else: downsized_tensor_ref = torch.load(f'./downsized_data/mask_times_attention_{animal_name}/{name}_{step_index}.pt').half().to('cuda')

        # get loss
        # ! k = 1 or k = 2?
        module_attn_loss = calculate_motion_rank_with_mask(attn_prob_example.detach(), attn_prob_control, 2, mask_example_input, downsized_tensor_ref)
        temp_attn_prob_loss.append(module_attn_loss)
            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each
    return loss_temp.mean()

def pad_to_target(tensor, target_height, target_width, padding_value=0):
    # Get the current shape of the input tensor
    shape = tensor.shape
    
    # Ensure the tensor has at least two dimensions
    if len(shape) < 2:
        raise ValueError("Input tensor must have at least two dimensions")
    
    # Calculate the required padding for height and width
    pad_height = max(target_height - shape[0], 0)
    pad_width = max(target_width - shape[1], 0)
    
    # Calculate the padding for each dimension
    # Since padding should be distributed evenly, we divide the padding between before and after
    pad_height_before = pad_height // 2
    pad_height_after = pad_height - pad_height_before
    pad_width_before = pad_width // 2
    pad_width_after = pad_width - pad_width_before
    
    # Create the padding tuple
    pad = (0, 0) * (len(shape) - 2) + (pad_width_before, pad_width_after, pad_height_before, pad_height_after)
    
    # Apply padding
    padded_tensor = F.pad(tensor, pad, mode='constant', value=padding_value)
    
    return padded_tensor

def calculate_motion_rank_with_mask(tensor_ref, tensor_gen, rank_k, mask_example_input, downsized_tensor_ref=None):

    # tensor shape         (HW, n_head, F, F)
    # mask input shape     (HW, F, 1) --> (HW, 1, F, 1)
                
    # TODO: fix hard-coded elements
    # if downsized reference is given, pad it and use it
    target_hw, _, _, _ = current_shape = tensor_ref.shape
    target_h = target_w = int(target_hw**0.5); assert target_h*target_w == target_hw 
    if downsized_tensor_ref is not None:
        hw, _, _, _ = downsized_shape = downsized_tensor_ref.shape
        h = w = int(hw**0.5); assert h*w == hw
        downsized_tensor_ref = downsized_tensor_ref.view(h, w, *tuple(downsized_shape[1:]))
        downsized_tensor_ref = pad_to_target(downsized_tensor_ref, target_h, target_w)
        tensor_ref = downsized_tensor_ref.view(current_shape)

    else:
        # reshape & binarize
        mask_example_input = (mask_example_input.unsqueeze(dim=1) > 0)
        tensor_ref *= mask_example_input

    # 기존 코드

    if rank_k == 0:
        loss = torch.tensor(0.0,device = tensor_ref.device)
    elif rank_k > tensor_ref.shape[-1]:
        raise ValueError("the value of rank_k cannot larger than the number of frames")
    else:   
        # sort tensor in ascending order
        _, sorted_indices = torch.sort(tensor_ref, dim=-1)

        # zeros of shape [64, 8, 16, 16-rank] + ones of shape [64, 8, 16, rank]
        # zeros: values to be masked,  ones: top-k values
        mask_indices = torch.cat((torch.zeros([*tensor_ref.shape[:-1],tensor_ref.shape[-1]-rank_k], dtype=torch.bool), torch.ones([*tensor_ref.shape[:-1],rank_k], dtype=torch.bool)),dim=-1)
        
        # retrieve the maximum values from reference tensor, expand it from [64, 8, 16, 1] --> [64, 8, 16, 16]
        max_copy = sorted_indices[:,:,:,[-1]].expand(*tensor_ref.shape[:-1], tensor_ref.shape[-1])
        sorted_indices[~mask_indices] = max_copy[~mask_indices]
        
        # create actual mask based on indices
        mask = torch.zeros_like(tensor_ref, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, True) 

        # make loss zero for masked-out region
        flat_ref = tensor_ref[mask].detach()
        flat_gen = tensor_gen[mask]
        nonzero_mask = (flat_ref != 0)

        # compute mse loss
        loss = F.mse_loss(flat_ref[nonzero_mask], flat_gen[nonzero_mask])
        
    return loss


# * semantic loss - shift ver
def compute_semantic_loss_with_shift( temp_attn_key_app, temp_attn_key_control, weight_each,mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type="temp"):
    temp_attn_key_loss = []
    for name in temp_attn_key_app.keys():

        attn_key_app = temp_attn_key_app[name]
        attn_key_control = temp_attn_key_control[name]
        if mask_example_fore == None:
            module_attn_loss = calculate_semantic_loss_with_shift(attn_key_app.detach(), attn_key_control, 
                                                    None, None, None, None, block_type)
        else:
            block_name = ".".join(name.split(".")[:2])
            module_attn_loss = calculate_semantic_loss_with_shift(attn_key_app.detach(), attn_key_control, 
                                                        mask_example_fore[block_name], mask_example_back[block_name],mask_app_fore[block_name], mask_app_back[block_name],block_type)

        temp_attn_key_loss.append(module_attn_loss)
            
    loss_app = torch.stack(temp_attn_key_loss) * weight_each
    return loss_app.mean()

def calculate_semantic_loss_with_shift(tensor_app, tensor_gen, mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type):
 
    if mask_example_fore is None:
        if block_type == "temp":
            # loss = F.mse_loss(tensor_app.mean(dim=0).detach(), tensor_gen.mean(dim=0))
            loss = F.mse_loss(tensor_app.mean(dim=[0,1],keepdim=True).detach(), tensor_gen.mean(dim=0,keepdim=True))  
        else:
            # loss = F.mse_loss(tensor_app.mean(dim=1).detach(), tensor_gen.mean(dim=1))  
            loss = F.mse_loss(tensor_app.mean(dim=[0,1], keepdim=True).detach(), tensor_gen.mean(dim=1,keepdim=True))  
 
    else:
        if block_type == "temp":
            tensor_app = tensor_app.permute(1,0,2)
            tensor_gen = tensor_gen.permute(1,0,2)
        # [frame, H*W, head*dim]

        ############

        # save original mask
        # to_save = mask_app_fore[0, :, 0].view(8,8)
        # to_pil = transforms.ToPILImage()
        # image = to_pil(to_save)
        # image.save('./masks/mask.png')

        # shift mask
        # mask shape: (F, HW, B=1)

        mask_app_fore = shift_mask(mask_app_fore)
        mask_app_back = 1 - mask_app_fore

        # gaussian_blur = get_gaussian_blur(mask_no_blur,sigma=5).to(mask_no_blur.dtype)
        # mask_example_foreground[block_name] = gaussian_blur* mask_no_blur
        # mask_example_background[block_name] = (1-gaussian_blur)*(1-mask_no_blur) 

        # save shifted mask
        # to_save = mask_app_fore[0, :, 0].view(8,8)
        # to_pil = transforms.ToPILImage()
        # image = to_pil(to_save)
        # image.save('./masks/shifted_mask_fore.png')

        ############

        ref_foreground = (tensor_app*mask_app_fore).sum(dim=1)/(mask_app_fore.sum(dim=1)+1e-5)
        ref_background = (tensor_app*mask_app_back).sum(dim=1)/(mask_app_back.sum(dim=1)+1e-5)
        
        gen_foreground = (tensor_gen*mask_example_fore).sum(dim=1)/(mask_example_fore.sum(dim=1)+1e-5)
        gen_background = (tensor_gen*mask_example_back).sum(dim=1)/(mask_example_back.sum(dim=1)+1e-5)
        
        loss =  F.mse_loss(ref_foreground.detach(), gen_foreground) +  F.mse_loss(ref_background.detach(), gen_background) 

    return loss



# * semantic loss - resize ver
def compute_semantic_loss_with_resize( temp_attn_key_app, temp_attn_key_control, weight_each,mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type="temp"):
    temp_attn_key_loss = []
    for name in temp_attn_key_app.keys():

        attn_key_app = temp_attn_key_app[name]
        attn_key_control = temp_attn_key_control[name]
        if mask_example_fore == None:
            module_attn_loss = calculate_semantic_loss_with_resize(attn_key_app.detach(), attn_key_control, 
                                                    None, None, None, None, block_type)
        else:
            block_name = ".".join(name.split(".")[:2])
            module_attn_loss = calculate_semantic_loss_with_resize(attn_key_app.detach(), attn_key_control, 
                                                        mask_example_fore[block_name], mask_example_back[block_name],mask_app_fore[block_name], mask_app_back[block_name],block_type)

        temp_attn_key_loss.append(module_attn_loss)
            
    loss_app = torch.stack(temp_attn_key_loss) * weight_each
    return loss_app.mean()

def calculate_semantic_loss_with_resize(tensor_app, tensor_gen, mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type):
 
    if mask_example_fore is None:
        if block_type == "temp":
            # loss = F.mse_loss(tensor_app.mean(dim=0).detach(), tensor_gen.mean(dim=0))
            loss = F.mse_loss(tensor_app.mean(dim=[0,1],keepdim=True).detach(), tensor_gen.mean(dim=0,keepdim=True))  
        else:
            # loss = F.mse_loss(tensor_app.mean(dim=1).detach(), tensor_gen.mean(dim=1))  
            loss = F.mse_loss(tensor_app.mean(dim=[0,1], keepdim=True).detach(), tensor_gen.mean(dim=1,keepdim=True))  
 
    else:
        if block_type == "temp":
            tensor_app = tensor_app.permute(1,0,2)
            tensor_gen = tensor_gen.permute(1,0,2)
        # [frame, H*W, head*dim]

        ############

        # shift mask
        # mask shape: (F, HW, B=1)

        # extract dim (res = h = w)
        f, hw, b = mask_app_fore.shape
        res = int(hw**0.5)

        # mask looks like the padded attention map
        # * F.interpolate() and pad_tensor() expectes input to have shape (f, c, h, w)
        new_mask_app_fore = torch.ones_like(mask_app_fore).view(f, 1, res, res)
        new_mask_app_fore_downsized = F.interpolate(new_mask_app_fore, size=(res//2, res//2), mode='nearest')
        new_mask_app_fore = pad_tensor(new_mask_app_fore_downsized, target_size=(f,1,res,res), mode=pad_mode)

        # reshape back 
        mask_app_fore = new_mask_app_fore.view(f, hw, b)
        mask_app_back = 1 - mask_app_fore

        # save shifted mask
        # to_save = new_mask_app_fore[0, :, 0].view(res, res)
        # to_pil = transforms.ToPILImage()
        # image = to_pil(to_save)
        # image.save('./masks/shifted_mask_fore.png')

        ############

        ref_foreground = (tensor_app*mask_app_fore).sum(dim=1)/(mask_app_fore.sum(dim=1)+1e-5)
        ref_background = (tensor_app*mask_app_back).sum(dim=1)/(mask_app_back.sum(dim=1)+1e-5)
        
        gen_foreground = (tensor_gen*mask_example_fore).sum(dim=1)/(mask_example_fore.sum(dim=1)+1e-5)
        gen_background = (tensor_gen*mask_example_back).sum(dim=1)/(mask_example_back.sum(dim=1)+1e-5)
        
        loss =  F.mse_loss(ref_foreground.detach(), gen_foreground) +  F.mse_loss(ref_background.detach(), gen_background) 

    return loss

def compute_CA_loss(cross_attn2_prob):
    
    # TODO: implement BCE loss instead of MSE 

    # cross_attn_blocks = ['down_blocks.1','down_blocks.2', 'up_blocks.1', 'up_blocks.2']
    cross_attn_blocks = ['up_blocks.1', 'up_blocks.2']
    token_index = (1, )
    loss = 0
    n_head = 8
    
    for i, block_name in enumerate(cross_attn_blocks):
            
        attn2_prob_example = []
        attn2_prob_app = []
        cross_attn2_prob_each_block = {key: cross_attn2_prob[key] for key in cross_attn2_prob.keys() if block_name in key}
        
        # collect all maps
        for name in cross_attn2_prob_each_block.keys():
            cross_attn2_prob_each = cross_attn2_prob_each_block[name]
            
            # 2: one for example, one for app,  8: n_heads
            double_f_nhead, hw, _= cross_attn2_prob_each.shape         
            f = double_f_nhead // (2 * n_head)                        
            cross_attn2_prob_each = cross_attn2_prob_each.view(2, f, n_head, hw, 77)
            
            # each has shape [f=16, head, h*w, 77]
            attn2_prob_example.append(cross_attn2_prob_each[0])
            attn2_prob_app.append(cross_attn2_prob_each[1]) 
            
        # concat in head dim
        attn2_prob_example = torch.cat(attn2_prob_example, dim = 1)
        attn2_prob_app = torch.cat(attn2_prob_app, dim = 1)
        
        # --> [b, h*w, 77]  
        # NOTE: b = batch_size = [ f(=16) * n_heads_(=8) * num_attn_blocks(= 2 or 3) ]
        attn2_prob_example = attn2_prob_example.view(-1, hw, 77)
        attn2_prob_app = attn2_prob_app.view(-1, hw, 77)
        
        for j in token_index:
            # --> [b, hw]
            attn2_prob_example_slice = attn2_prob_example[:,:,j]
            attn2_prob_app_slice = attn2_prob_app[:,:,j]

            # soft threshold
            # attn2_prob_example_slice = soft_threshold(attn2_prob_example_slice, dim=1, s=10)
            # attn2_prob_app_slice = soft_threshold(attn2_prob_app_slice, dim=1, s=10)

            # hard threhsold
            attn2_prob_example_slice = hard_threshold(attn2_prob_example_slice, threshold=0.3)
            attn2_prob_app_slice = hard_threshold(attn2_prob_app_slice, threshold=0.3)

            # compute BCE betweem two
            # loss +=  5 * F.mse_loss(input=attn2_prob_app_slice, target=attn2_prob_example_slice)
            loss +=  0.05 * F.binary_cross_entropy(input=attn2_prob_app_slice, target=attn2_prob_example_slice) 


    return loss

# * background perservation loss
def compute_background_loss(self_attn_prob, mask_example_fore, step_index):
    # TODO: implement BCE loss instead of MSE 

    self_attn_blocks = ['up_blocks.1', 'up_blocks.2']
    # self_attn_blocks = ['up_blocks.1']
    n_head = 8
    loss = 0
    for i, block_name in enumerate(self_attn_blocks):
            
        attn2_prob_example = []
        attn2_prob_app = []
        self_attn_prob_each_block = {key: self_attn_prob[key] for key in self_attn_prob.keys() if block_name in key}
        
        # collect all maps
        for name in self_attn_prob_each_block.keys():
            self_attn_prob_each = self_attn_prob_each_block[name]
            
            # reshape (NOTE: b = 2 * n_head * f)
            b, hw, hw = self_attn_prob_each.shape
            self_attn_prob_each = self_attn_prob_each.view(2, b//2, hw, hw)
            
            # sanity check
            f = b // (2 * n_head)
            assert b == 2 * f * n_head

            # each has shape [b//2, h*w, h*w]
            attn2_prob_example.append(self_attn_prob_each[0].detach())
            attn2_prob_app.append(self_attn_prob_each[1]) 
            
        # concat in batch dim: --> [B, h*w, h*w], B = b * # layers 
        attn2_prob_example = torch.cat(attn2_prob_example, dim = 0)
        attn2_prob_app = torch.cat(attn2_prob_app, dim = 0)

        # reshape
        attn2_prob_example = attn2_prob_example.view(-1, 16, hw, hw)
        attn2_prob_app = attn2_prob_app.view(-1, 16, hw, hw)

        # binarize & reshape mask --> (1, f, hw, 1) for broadcasting
        mask_background_each = (1 - mask_example_fore[block_name]).unsqueeze(dim=0)

        # extract background
        attn2_prob_example *= mask_background_each
        attn2_prob_app *= mask_background_each

        # exclude zero elements
        nonzero_mask = (attn2_prob_example != 0) & (attn2_prob_app != 0)
        loss += 100 * F.l1_loss(input=attn2_prob_app[nonzero_mask], target=attn2_prob_example[nonzero_mask])
              
    return loss


# * attention value loss
def compute_value_loss(temp_attn_value_example, temp_attn_value_control, transformation=None, resize_factor = None, shift_dir=None):
    
    temp_attn_key_loss = []

    for name in temp_attn_value_example.keys():

        attn_value_example = temp_attn_value_example[name].detach()
        attn_value_control = temp_attn_value_control[name]

        hw, f, dim = attn_value_example.shape
        h = w = int(hw**0.5); assert h*w == hw

        # ? resize
        resize = True if transformation == 'resize' else False
        if resize:
            scale_factor = resize_factor; s = scale_factor ** 0.5
            attn_value_example = attn_value_example.view(h, w, f, dim)
            # attn_value_example_copy = attn_value_example.clone()

            # --> (f, dim, h, w)
            # --> (f, dim, h', w')
            attn_value_example = attn_value_example.permute(2, 3, 0, 1)
            attn_value_example = F.interpolate(attn_value_example, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
            
            # --> (f, dim, h, w)
            if scale_factor <= 1:
                attn_value_example = pad_tensor(attn_value_example, target_size=(f, dim, h, w), mode=pad_mode) 
                # attn_value_example = attn_value_example + (attn_value_example==0)*attn_value_example_copy.permute(2,3,0,1)
            else:
                attn_value_example = crop_tensor(attn_value_example, target_size=(f, dim, h, w), mode=pad_mode)

            # --> (h, w, f, dim)
            attn_value_example = attn_value_example.permute(2, 3, 0, 1)

            # --> (hw, f, dim)
            if scale_factor <= 1:
                attn_value_example = attn_value_example.view(hw, f, dim)
            else:
                attn_value_example = attn_value_example.reshape(hw, f, dim)


        # ? original flip
        flip = True if transformation == 'flip' else False
        if flip:
            attn_value_example = attn_value_example.view(h, w, f, dim)
            attn_value_example = attn_value_example.flip(dims=[1])
            attn_value_example = attn_value_example.view(hw, f, dim)
        

        # ? flip(?) new
        flip_new = True if transformation == 'flip_new' else False
        if flip_new:
            attn_value_example = attn_value_example.view(h, w, f, dim)
            attn_value_control = attn_value_control.view(h, w, f, dim)

            #
            # Compute displacement vectors as before, but ensuring tensors track gradients
            diff_feature_1_ex = attn_value_example[:, :, 1, :] - attn_value_example[:, :, 0, :]  # Difference for frames 1 and 2
            diff_feature_15_ex = attn_value_example[:, :, 15, :] - attn_value_example[:, :, 14, :]  # Difference for frames 15 and 16
            mean_diff_feature_1_ex = torch.sum(diff_feature_1_ex, dim=-1)  # Shape (h, w)
            mean_diff_feature_15_ex = torch.sum(diff_feature_15_ex, dim=-1)  # Shape (h, w)
            centroid_1_ex = compute_centroid(mean_diff_feature_1_ex)
            centroid_15_ex = compute_centroid(mean_diff_feature_15_ex)
            displacement_vector_ex = torch.stack([centroid_15_ex[0] - centroid_1_ex[0], centroid_15_ex[1] - centroid_1_ex[1]])

            diff_feature_1_control = attn_value_control[:, :, 1, :] - attn_value_control[:, :, 0, :]  # Difference for frames 1 and 2
            diff_feature_15_control = attn_value_control[:, :, 15, :] - attn_value_control[:, :, 14, :]  # Difference for frames 15 and 16
            mean_diff_feature_1_control = torch.sum(diff_feature_1_control, dim=-1)  # Shape (h, w)
            mean_diff_feature_15_control = torch.sum(diff_feature_15_control, dim=-1)  # Shape (h, w)
            centroid_1_control = compute_centroid(mean_diff_feature_1_control)
            centroid_15_control = compute_centroid(mean_diff_feature_15_control)
            displacement_vector_control = torch.stack([centroid_15_control[0] - centroid_1_control[0], centroid_15_control[1] - centroid_1_control[1]])

            # normalize
            displacement_vector_ex_norm = displacement_vector_ex / (torch.norm(displacement_vector_ex) + 1e-5)
            displacement_vector_control_norm = displacement_vector_control / (torch.norm(displacement_vector_control) + 1e-5)

            print(displacement_vector_ex_norm)
            print(displacement_vector_control_norm)

            # Reshape to be 1x2 tensors for compatibility with F.cosine_similarity
            displacement_vector_ex_norm = displacement_vector_ex_norm.unsqueeze(0)  # Shape (1, 2)
            displacement_vector_control_norm = displacement_vector_control_norm.unsqueeze(0)  # Shape (1, 2)

            # Compute cosine similarity
            module_attn_loss = 0.1 * (F.mse_loss(displacement_vector_control_norm, displacement_vector_ex_norm))
            

            attn_value_example = attn_value_example.view(hw, f, dim)
            attn_value_control = attn_value_control.view(hw, f, dim)


        # ? shift
        shift =  True if transformation == 'shift' else False
        if shift:

            if shift_dir == 'up':
                shift_by = (h//5, 0)

            elif shift_dir == 'left':
                shift_by = (0, w//4)

            elif shift_dir == 'down':
                shift_by = (-h//5, 0)
            
            elif shift_dir == 'right':
                shift_by = (0, -w//6)

            attn_value_example = shift_tensor(attn_value_example.unsqueeze(dim=-1), shift=shift_by).view(hw, f, dim)
        

        # ? rotate
        rotate = True if transformation == 'rotate' else False
        if rotate:
            attn_value_example = attn_value_example.view(h, w, f, dim)
            attn_value_example = rotate_tensor(attn_value_example, angle=-30.0)
            attn_value_example = attn_value_example.view(hw, f, dim)
        # ? 

        # ? shear
        shear = True if transformation == 'shear' else False
        if shear:
            attn_value_example = attn_value_example.view(h, w, f, dim)
            attn_value_example = apply_shear_transform(attn_value_example, shear_factor=0.2)
            attn_value_example = attn_value_example.view(hw, f, dim)


        # ? perspective warp
        warp = True if transformation == 'warp' else False
        if warp:
            if hw==100:
                d = 2

                if shift_dir == 'right':
                    dst_pts = [(0, 0), (9, 0), (9, 9), (0, 9)]
                    src_pts = [(0, 0), (9, d), (9, 9-d), (0, 9)]
                elif shift_dir == 'left':
                    dst_pts = [(0, 0), (9, 0), (9, 9), (0, 9)]
                    src_pts = [(0, d), (9, 0), (9, 9), (0, 9 - d)]
                elif shift_dir == 'down':
                    dst_pts = [(0, 0), (9, 0), (9, 9), (0, 9)]
                    src_pts = [(0, 0), (9, 0), (9-d, 9-d), (d, 9 - d)]

            elif hw==400:
                d = 4
                if shift_dir == 'right':
                    dst_pts = [(0, 0), (19, 0), (19, 19), (0, 19)]
                    src_pts = [(0, 0), (19, d), (19, 19-d), (0, 19)]
                elif shift_dir == 'left':
                    dst_pts = [(0, 0), (19, 0), (19, 19), (0, 19)]
                    src_pts = [(0, d), (19, 0), (19, 19), (0, 19 - d)]
                elif shift_dir == 'down':
                    dst_pts = [(0, 0), (19, 0), (19, 19), (0, 19)]
                    src_pts = [(0, 0), (19, 0), (19-d, 19-d), (d, 19 - d)]


            dst_pts = torch.tensor(dst_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)
            src_pts = torch.tensor(src_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)

            # Get the homography matrix
            H = kornia.geometry.get_perspective_transform(src_pts, dst_pts).float()

            # Reshape and perform the custom perspective warp
            attn_value_example = attn_value_example.view(h, w, f, dim)
            attn_value_example = attn_value_example.permute(2, 3, 0, 1).view(f * dim, 1, h, w)

            # Prepare homography matrix for the batch
            H_batch = H.repeat(f * dim, 1, 1)  

            # Apply the custom perspective warp with tolerance
            # print(attn_value_example.shape)
            for _ in range(int(resize_factor)):
                attn_value_example = kornia.geometry.warp_perspective(attn_value_example.float(), H_batch, dsize=(h,w), mode='bicubic', align_corners=True)

            # Reshape the result back to the original dimensions
            attn_value_example = attn_value_example.view(f, dim, h, w)
            attn_value_example = attn_value_example.permute(2, 3, 0, 1).view(hw, f, dim).half()


        # ? random sample
        random_sample = True if transformation == 'random_sample' else False
        if random_sample:
            sample_rate_dim, sample_rate_hw = 0.1, 0.1 # this depends on the extent of camera zoom...
            num_samples_dim = int(dim * sample_rate_dim)
            num_samples_hw = int(hw * sample_rate_hw)

            # Calculate indices for sampling
            sampled_indices_dim = torch.linspace(0, dim - 1, steps=num_samples_dim).long()
            sampled_indices_hw = torch.linspace(0, hw - 1, steps=num_samples_hw).long()

            #
            attn_value_example = attn_value_example[..., sampled_indices_dim]
            attn_value_example = attn_value_example[sampled_indices_hw, ...]

            attn_value_control = attn_value_control[..., sampled_indices_dim]
            attn_value_control = attn_value_control[sampled_indices_hw, ...]

            # Apply these sampled indices to the tensors
            module_attn_loss = F.mse_loss(attn_value_control, attn_value_example)
            temp_attn_key_loss.append(module_attn_loss)
            continue

        # print(attn_value_example.mean(), attn_value_control.mean())
        # print(attn_value_example.std(), attn_value_control.std())

        # *
        nonzero_mask = attn_value_example != 0
        module_attn_loss = F.mse_loss(input=attn_value_control[nonzero_mask], target=attn_value_example[nonzero_mask])
        # module_attn_loss = F.mse_loss(input=attn_value_control, target=attn_value_example)

        if hw==100: module_attn_loss = 4 * module_attn_loss
        temp_attn_key_loss.append(module_attn_loss)
            

    loss_value = torch.stack(temp_attn_key_loss) 

    return 2*loss_value.mean()



# * save M_ * A_
def save_mask_and_attention_map(temp_attn_prob_example, mask_example_fore, step_index, animal_name=None):
    
    savedir_mask = f'./downsized_data/mask_{animal_name}'
    savedir_MA = f'./downsized_data/mask_times_attention_{animal_name}'
    if not os.path.exists(savedir_mask): os.makedirs(savedir_mask)
    if not os.path.exists(savedir_MA): os.makedirs(savedir_MA)

    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        # retrieve mask
        mask_key = f'up_blocks.{block_num}'
        mask_example_input = mask_example_fore[mask_key]
        mask_example_input = rearrange(mask_example_input, 'f hw b -> hw f b') # b=1

        # save mask at t=250
        hw, f, _ = mask_example_input.shape 
        h = w = int(hw**0.5)
        if step_index == 250 - 1:   
            for i in range(0, f):
                mask_slice = mask_example_input[:, i, 0].view(h, w).detach().cpu()
                plt.figure(figsize=(8, 6))  
                plt.imshow(mask_slice, cmap='viridis', vmin=0.0, vmax=1.0)  
                plt.colorbar() 
                plt.savefig(f'./downsized_data/mask_{animal_name}/mask_{mask_key}_{i:03d}.png')

        # retrieve attention maps
        attn_prob_example = temp_attn_prob_example[name]

        # compute M_ * A_
        mask_example_input = (mask_example_input.unsqueeze(dim=1) > 0)
        attn_prob_example *= mask_example_input
    
        # save
        save_path = f"./downsized_data/mask_times_attention_{animal_name}/{name}_{step_index}.pt"
        torch.save(attn_prob_example, save_path)



# ! deprecated
# ! ######################

# * location loss
def compute_location_loss(mask_example_fore, mask_app_fore, loss_type = "MSE"):

    location_loss_list = []
    names = mask_example_fore.keys()

    device, dtype = 'cuda:0', torch.float16

    for name in names:

        # initial shape: (F, HW, B=1)
        mask_example_fore[name] = shift_mask(mask_example_fore[name]).squeeze(-1).detach()
        mask_app_fore[name] = mask_app_fore[name].squeeze(-1)

        if loss_type == "MSE":
            loss = F.mse_loss(mask_example_fore[name], mask_app_fore[name])

        elif loss_type == "CE":
            loss = F.binary_cross_entropy(mask_app_fore[name], mask_example_fore[name])

        else:
            raise NotImplementedError
        
        location_loss_list.append(loss)

    loss = torch.mean(torch.stack(location_loss_list) * torch.tensor([6, 4], dtype=dtype, device=device))

    return loss

# * size loss
def compute_size_loss_cross(cross_attn2_prob, size_factor = 0.5):
    
    # TODO: change hard-coded elements
    # ? up_blocks.3를 포함하면 nan이 뜸 
    cross_attn_blocks = ['down_blocks.1','down_blocks.2', 'up_blocks.1', 'up_blocks.2']
    token_index_app, token_index_example = 1, 1
    loss, loss_scale = 0, 10
    n_head = 8
    
    for i, block_name in enumerate(cross_attn_blocks):
            
        attn2_prob_example = []
        attn2_prob_app = []
        corss_attn2_prob_each_block = {key: cross_attn2_prob[key] for key in cross_attn2_prob.keys() if block_name in key}
        
        # collect all maps
        for name in corss_attn2_prob_each_block.keys():
            cross_attn2_prob_each = corss_attn2_prob_each_block[name]
            
            # 2: one for example, one for app,  8: n_heads
            double_f_nhead, hw, _= cross_attn2_prob_each.shape         
            f = double_f_nhead // (2 * n_head)                        
            cross_attn2_prob_each = cross_attn2_prob_each.reshape(2,f,-1,cross_attn2_prob_each.shape[1],cross_attn2_prob_each.shape[2])
            
            # each has shape [f=16, head, h*w, 77]
            attn2_prob_example.append(cross_attn2_prob_each[1])
            attn2_prob_app.append(cross_attn2_prob_each[0]) 
            
        # concat in head dim
        attn2_prob_example = torch.cat(attn2_prob_example, dim = 1)
        attn2_prob_app = torch.cat(attn2_prob_app, dim = 1)
        
        # --> [b, h*w, 77]  
        # NOTE: b = batch_size = [ f(=16) * n_heads_(=8) * num_attn_blocks(= 2 or 3) ]
        attn2_prob_example = attn2_prob_example.view(-1, hw, 77).detach()
        attn2_prob_app = attn2_prob_app.view(-1, hw, 77)
        
        # --> [b, hw, 1] --> [b, hw]
        attn2_prob_example_slice = attn2_prob_example[:,:,[token_index_example]].squeeze()
        attn2_prob_app_slice = attn2_prob_app[:,:,[token_index_app]].squeeze()

        # ? attention map visualization
        # h = int(hw**0.5)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(attn2_prob_example_slice[0, :, 0].view(h, h).cpu(), cmap='viridis')  
        # plt.colorbar(); plt.show()

        # soft thresholding
        attn2_prob_example_slice = soft_threshold(attn2_prob_example_slice, dim=1, s=50)
        attn2_prob_app_slice = soft_threshold(attn2_prob_app_slice, dim=1, s=50)

        # compute size for each batch element --> (b, )
        hw = attn2_prob_example_slice.shape[1]
        example_size = torch.sum(attn2_prob_example_slice, dim=1) / hw
        app_size = torch.sum(attn2_prob_app_slice, dim=1) / hw

        print(f"example size: {torch.mean(example_size).item() :.3f}")
        print(f"app_size    : {torch.mean(app_size).item() : .3f}")

        # find average loss across frames
        target_size = example_size * size_factor
        loss += loss_scale * ( torch.mean(F.l1_loss(target_size, app_size)) ) 

    return loss

def compute_size_loss_temp(temp_attn_prob_example, temp_attn_prob_control, size_factor = 0.5):
    
    loss = 0
    example_size_list, app_size_list = [], []

    # TODO: change hard-coded elements
    for name in temp_attn_prob_example.keys():
            
        # extract slice
        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name].detach()

        # dim setup
        hw, n_head, f, f = attn_prob_example.shape
        h = w = int(hw**0.5)

        # mean over heads
        # --> (hw, f, f)
        attn_prob_example = torch.mean(attn_prob_example, dim=1, keepdim=False)
        attn_prob_control = torch.mean(attn_prob_control, dim=1, keepdim=False)

        # soft-threshold
        attn_prob_example = soft_threshold(attn_prob_example, dim=0)
        attn_prob_control = soft_threshold(attn_prob_control, dim=0)

        # reshape
        # --> (hw, f*f)
        attn_prob_example = attn_prob_example.view(h*w, f*f)
        attn_prob_control = attn_prob_control.view(h*w, f*f)

        # compute size
        example_size = torch.sum(attn_prob_example, dim=0, keepdim=False) / hw
        app_size = torch.sum(attn_prob_control, dim=0, keepdim=False) / hw

        example_size_list.append(torch.mean(example_size).item())
        app_size_list.append(torch.mean(app_size).item())

        # find average loss over frames
        target_size = example_size * size_factor
        loss += (  torch.mean(F.l1_loss(target_size, app_size))  )

    print(f"example size: {sum(example_size_list) / len(example_size_list) :.3f}")
    print(f"app_size    : {sum(app_size_list) / len(app_size_list) : .3f}")

    return loss

def compute_size_loss_temp_v2(temp_attn_prob_example, temp_attn_prob_control, size_factor = 0.5, step_index=None):
    
    loss = 0
    example_size_list, control_size_list = [], []

    for name in temp_attn_prob_example.keys():
            
        # extract slice
        attn_prob_example = temp_attn_prob_example[name].detach()
        attn_prob_control = temp_attn_prob_control[name]

        # dim setup
        hw, n_head, f, f = attn_prob_example.shape
        h = w = int(hw**0.5)

        # reshape
        # --> (b, f, f, hw)
        attn_prob_example = attn_prob_example.view(n_head, f, f, hw)
        attn_prob_control = attn_prob_control.view(n_head, f, f, hw)

        # plot
        if step_index in [50, 150, 250] and hw == 100:
            tensor, _ = attn_prob_example.max(dim=0, keepdim=False)
            tensor = tensor.view(f, f, h, w)

            # Initialize an array to hold the selected heatmaps
            selected_heatmaps = []

            # Loop through each row in the 2D grid to find the heatmap with the largest sum
            for i in range(f):

                # rough thresholding
                m = tensor[i] > 0.5
                #

                row_sums = (tensor*m)[i].sum(dim=(1, 2))  # Sum across h and w dimensions
                max_index = torch.argmax(row_sums)  # Find the index of the max sum
                selected_heatmap = tensor[i, max_index]  # Select the heatmap with the max sum
                selected_heatmaps.append(selected_heatmap)  # Append to the list of selected heatmaps

            # Convert the selected heatmaps list to a tensor
            selected_heatmaps = torch.stack(selected_heatmaps)

            # Plot the selected heatmaps as a 1D plot with f subplots
            fig, axs = plt.subplots(1, f, figsize=(4 * f, 4))

            for i in range(f):
                ax = axs[i] if f > 1 else axs
                cax = ax.imshow(selected_heatmaps[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
                ax.set_title(f'Slice {i+1}')
                fig.colorbar(cax, ax=ax)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.savefig(f'./Heatmaps/t={step_index}.png')
            plt.close()

        # self-temp attn only
        # --> (b, f, hw)
        attn_prob_example = select_matching_indices(attn_prob_example)
        attn_prob_control = select_matching_indices(attn_prob_control)


        # threshold
        attn_prob_example = soft_threshold(attn_prob_example, dim=-1) 
        attn_prob_control = soft_threshold(attn_prob_control, dim=-1)

        # compute size
        # --> (b, f)
        example_size = torch.sum(attn_prob_example, dim=-1, keepdim=False) / hw * size_factor
        control_size = torch.sum(attn_prob_control, dim=-1, keepdim=False) / hw

        # find average loss over (b, f)
        loss += 3 * F.mse_loss(input=control_size, target=example_size)

    # print(f"example size: {sum(example_size_list) / len(example_size_list) :.5f}")
    # print(f"control_size: {sum(control_size_list) / len(control_size_list) :.5f}")

    return loss

def compute_notsize_loss_temp_v2(temp_attn_prob_example, temp_attn_prob_control, size_factor = 0.5, step_index=None):
    
    loss = 0
    example_size_list, control_size_list = [], []

    for name in temp_attn_prob_example.keys():
            
        # extract slice
        attn_prob_example = temp_attn_prob_example[name].detach()
        attn_prob_control = temp_attn_prob_control[name]

        # dim setup
        hw, n_head, f, f = attn_prob_example.shape
        h = w = int(hw**0.5)

        # heatmap loss
        tensor_example = attn_prob_example.view(n_head, f, f, h, w)
        tensor_control = attn_prob_control.view(n_head, f, f, h, w)

        #  ! plot
        if step_index == 150 and hw == 100:
            tensor, _ = tensor_example.max(dim=0, keepdim=False)

            # Initialize an array to hold the selected heatmaps
            selected_heatmaps = []

            # Loop through each row in the 2D grid to find the heatmap with the largest sum
            for i in range(f):

                # rough thresholding
                m = tensor[i] > 0.15
                #

                row_sums = (tensor*m)[i].sum(dim=(-1, -2))  # Sum across h and w dimensions
                max_index = torch.argmax(row_sums)  # Find the index of the max sum
                selected_heatmap = tensor[i, max_index]  # Select the heatmap with the max sum
                selected_heatmaps.append(selected_heatmap)  # Append to the list of selected heatmaps

            # Convert the selected heatmaps list to a tensor
            selected_heatmaps = torch.stack(selected_heatmaps)

            # Plot the selected heatmaps as a 1D plot with f subplots
            fig, axs = plt.subplots(1, f, figsize=(4 * f, 4))

            for i in range(f):
                ax = axs[i] if f > 1 else axs
                cax = ax.imshow(selected_heatmaps[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
                ax.set_title(f'Slice {i+1}')
                fig.colorbar(cax, ax=ax)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.savefig(f'./Heatmaps/t={step_index}.png')
            plt.close()
        # 


        k = 1  # Set k to 2 for top 2, or 3 for top 3, etc.
        selected_heatmaps_example = []
        selected_heatmaps_control = []

        for i in range(f):
            # Example tensor: Sum across h and w dimensions for each head
            m_example = tensor_example > 0.3
            row_sums_example = ((tensor_example * m_example)[:, i]).sum(dim=(-1, -2))  # Sum across h, w
            
            # Find the top k heatmaps in each batch (n_head) with the maximum sums for this row
            topk_indices_example = torch.topk(row_sums_example, k=k, dim=1).indices  # Get top k across the f dimension (within each head)
            
            # Gather the top k heatmaps for each batch and row
            selected_heatmaps_example.append(tensor_example[torch.arange(n_head).unsqueeze(-1), i, topk_indices_example])

            # Control tensor: Sum across h and w dimensions for each head
            m_control = tensor_control > 0.3
            row_sums_control = ((tensor_control * m_control )[:, i]).sum(dim=(-1, -2))  # Sum across h, w
            
            # Find the top k heatmaps in each batch (n_head) with the maximum sums for this row
            topk_indices_control = torch.topk(row_sums_control, k=k, dim=1).indices  # Get top k across the f dimension (within each head)
            
            # Gather the top k heatmaps for each batch and row
            selected_heatmaps_control.append(tensor_control[torch.arange(n_head).unsqueeze(-1), i, topk_indices_control])

        # Stack the selected heatmaps along the f dimension
        selected_heatmaps_example = torch.cat(selected_heatmaps_example, dim=1)  # Concatenate along the f dimension
        selected_heatmaps_control = torch.cat(selected_heatmaps_control, dim=1)  # Concatenate along the f dimension


        # Calculate the average loss over (n_head, f)
        nonzero_mask = (selected_heatmaps_example > 0.3 ) 
        loss += F.mse_loss(input=selected_heatmaps_control[nonzero_mask], target=selected_heatmaps_example[nonzero_mask])


    # print(f"example size: {sum(example_size_list) / len(example_size_list) :.5f}")
    # print(f"control_size: {sum(control_size_list) / len(control_size_list) :.5f}")

    return loss



