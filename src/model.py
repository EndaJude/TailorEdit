import os
from einops import rearrange
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.nn import functional as F

from types import MethodType
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file

from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPVisionModelWithProjection
from modules.pipeline import TailorEditPixPipeline

def load_pipeline(args, device):
    # load modules
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, local_files_only=True
    )
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_vision_model_name_or_path, subfolder="models/image_encoder", revision=args.revision, local_files_only=True
    )    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True
    )
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, local_files_only=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # # rebound relateive method for adapting ip-adapter
    from modules.unet import process_encoder_hidden_states_new
    unet.process_encoder_hidden_states = MethodType(process_encoder_hidden_states_new, unet)

    # # initiate ip-adapter and change the attention_processor
    from modules.unet import load_ip_adapter_weights
    ip_adapter_state_dict = torch.load(args.ip_adapter_path, map_location=torch.device('cpu'), weights_only=True)
    load_ip_adapter_weights(unet, ip_adapter_state_dict)
    
    # initiate LoRA
    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_rank,
            bias="none",
            target_modules=["attn1.to_k", "attn1.to_v", "attn1.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_q"],
        )
        unet = inject_adapter_in_model(lora_config, unet, "rep")
        unet = inject_adapter_in_model(lora_config, unet, "chg")
        unet = inject_adapter_in_model(lora_config, unet, "add")
        unet = inject_adapter_in_model(lora_config, unet, "del")

    from modules.unet import init_moe_in_lora
    init_moe_in_lora(unet)
        
    unet_states_dict = load_file(os.path.join(args.tailor_edit_path, "UNet2DConditionModel/diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(unet_states_dict)
    del unet_states_dict

    controlnet_states_dict = load_file(os.path.join(args.tailor_edit_path, "ControlNetModel/diffusion_pytorch_model.safetensors"))
    controlnet.load_state_dict(controlnet_states_dict)
    del controlnet_states_dict

    # Freeze modules
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    vision_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    controlnet.requires_grad_(False).eval()

    # transfer into gpu
    if device != "cpu":
        unet.to(device).eval()
        controlnet.to(device).eval()
        text_encoder.to(device).eval()
        vision_encoder.to(device).eval()
        vae.to(device).eval()

    pipeline = TailorEditPixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        text_encoder=text_encoder,
        vae=vae,
        image_encoder=vision_encoder,
        gg_path = args.global_gate_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=args.weight_dtype,
    )

    return pipeline

def inference(pipeline, ppn, args, data, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # log the result localy
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        # load data
        edit_image = data["image"].to(device)
        src_parsing = data["parsing"].to(device)
        refer_image = data["reference"].to(device) if data["reference"] is not None else None
        instruction = data["instruction"]
        # target parsing prediction
        tar_parsing, src_parsing_pal, tar_parsing_pal = ppn.inference(src_parsing, instruction)
        # import pdb;pdb.set_trace()
        tar_parsing = rearrange(F.one_hot(tar_parsing.long(), num_classes=18), "b h w c ->b c h w")    
        edited_image = pipeline(
                            instruction,
                            image=edit_image,
                            control_image=tar_parsing,
                            num_inference_steps=50,
                            image_guidance_scale=2,
                            guidance_scale=5,
                            generator=generator,
                            ip_adapter_image=refer_image,
                            activate_lora_expert=None
                        ).images
        transform = transforms.ToTensor()
        edited_image = torch.stack([transform(img) for img in edited_image])
        
        if refer_image is not None:
            # resize reference image
            height, width = edited_image.shape[-2:]
            refer_image = F.interpolate(refer_image, size=(height, width), mode='bilinear', align_corners=False)

        edited_array = (edited_image[0] * 255).cpu().numpy().astype(np.uint8)
        edit_array = ((edit_image[0] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
        src_parsing_array = src_parsing_pal.astype(np.uint8).transpose(2, 0, 1)
        tar_parsing_array = tar_parsing_pal.astype(np.uint8).transpose(2, 0, 1)

        if refer_image is not None:
            refer_array = ((refer_image[0] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
            combined = np.concatenate((edit_array, src_parsing_array, tar_parsing_array, refer_array, edited_array), axis=2).transpose(1, 2, 0)
        else:
            combined = np.concatenate((edit_array, src_parsing_array, tar_parsing_array, edited_array), axis=2).transpose(1, 2, 0)

        os.makedirs(f"{args.output_dir}/comp", exist_ok=True)
        os.makedirs(f"{args.output_dir}/res", exist_ok=True)
        
        combined_image = Image.fromarray(combined)
        output_dir = f"{args.output_dir}/comp/{instruction[0]}.jpg"
        combined_image.save(output_dir)

        result_pil = Image.fromarray(edited_array.transpose(1, 2, 0))
        result_pil.save(f"{args.output_dir}/res/{instruction[0]}.jpg")