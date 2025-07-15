import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchmetrics.image.fid import FrechetInceptionDistance
from types import MethodType

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionInstructPix2PixPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils.dataset import InstructFashionDataset
from utils.arguments import parse_args, parse_args_for_debug
from utils.utils import show_fid, draw_loss, update_fid
from modules.pipeline import TailorEditPixPipeline
# from modules.unet import get_edit_localization_loss

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["original_image", "target_parsing", "edited_image", "edit_prompt"]

fid_list = []
    
def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
    dataloader,
    steps,
    do_fid=True,
    num_fid=300
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # log the result localy
    log_dir = os.path.join(args.output_dir, args.logging_dir)
    val_dir = f"{log_dir}/val/{steps:07d}"
    os.makedirs(val_dir, exist_ok=True)
    num_validation = args.num_validation_images
    if do_fid:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    with autocast_ctx:
        for idx, batch in enumerate(dataloader):
            if idx * dataloader.batch_sampler.batch_size >= num_fid:
                break

            edit_image = batch["original_pixel_values"]
            parsing = batch["target_parsing_values"]
            refer_image = batch["reference_pixel_values"] if "reference_pixel_values" in batch else None
            instruction = batch["input_ids"]
            target_image = batch["edited_pixel_values"]
            edited_image = pipeline(
                                instruction,
                                image=edit_image,
                                control_image=parsing,
                                num_inference_steps=30,
                                image_guidance_scale=1.5,
                                guidance_scale=7,
                                generator=generator,
                                ip_adapter_image=refer_image
                            ).images
            transform = transforms.ToTensor()
            edited_image = torch.stack([transform(img) for img in edited_image])
            
            if do_fid:
                gt_images = [(img + 1) / 2 * 255 for img in target_image]
                gen_images = [img * 255 for img in edited_image]
                gt_tensors = torch.stack(gt_images)
                gen_tensors = torch.stack(gen_images)
                fid = update_fid(fid, gt_tensors, gen_tensors)
            
            # resize reference image
            if refer_image is not None:
                height, width = edited_image.shape[-2:]
                refer_image = F.interpolate(refer_image, size=(height, width), mode='bilinear', align_corners=False)

            batch_size = edited_image.shape[0]
            for idx in range(batch_size):
                num_validation -= 1
                if num_validation < 0:
                    break

                edited_array = (edited_image[idx] * 255).cpu().numpy().astype(np.uint8)
                edit_array = ((edit_image[idx] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
                target_array = ((target_image[idx] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
                
                if refer_image is not None:
                    refer_array = ((refer_image[idx] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
                    combined = np.concatenate((edit_array, refer_array, edited_array, target_array), axis=2).transpose(1, 2, 0)
                else:
                    combined = np.concatenate((edit_array, edited_array, target_array), axis=2).transpose(1, 2, 0)
                combined_image = Image.fromarray(combined)
                output_dir = f"{val_dir}/{instruction[idx]}.jpg"
                combined_image.save(output_dir)
    
    if do_fid:
        output_dir = os.path.join(args.output_dir, "fid.jpg")
        show_fid(fid, fid_list, output_dir)

def main():
    args = parse_args()
    
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, local_files_only=True
    )
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
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, local_files_only=True
    )
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path, local_files_only=True)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # rebound relateive method for adapting ip-adapter
    from modules.unet import process_encoder_hidden_states_new
    unet.process_encoder_hidden_states = MethodType(process_encoder_hidden_states_new, unet)

    # initiate ip-adapter and change the attention_processor
    from modules.unet import load_ip_adapter_weights
    ip_adapter_state_dict = torch.load(args.ip_adapter_path, map_location=torch.device('cpu'), weights_only=True)
    load_ip_adapter_weights(unet, ip_adapter_state_dict)
    
    # initiate LoRA
    from peft import LoraConfig, inject_adapter_in_model
    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_rank,
            bias="none",
            target_modules=["attn1.to_k", "attn1.to_v", "attn1.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_q"],
        )
        unet = inject_adapter_in_model(lora_config, unet, args.lora_task)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vision_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    from modules.unet import set_unet_weight_requires_grad_
    set_unet_weight_requires_grad_(unet)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, model.__class__.__name__))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    update_params = [param for param in unet.parameters() if param.requires_grad]
    print(len(update_params))
    # update_params.extend([param for param in controlnet.parameters() if param.requires_grad])
    
    optimizer = optimizer_cls(
        update_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        train_dataset = InstructFashionDataset(
            path=args.dataset_name,
            task=args.lora_task,
            split="train",
            min_resize_res=args.resolution,
            max_resize_res=args.resolution,
            flip_prob=args.random_flip,
        )
        val_dataset = InstructFashionDataset(
            path=args.dataset_name,
            task=args.lora_task,
            split="val",
            min_resize_res=args.resolution,
            max_resize_res=args.resolution,
            flip_prob=0,
        )
    else:
        raise ValueError("Not been completed.")
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    def collate_fn(examples):
        ems = dict()
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        ems["original_pixel_values"] = original_pixel_values

        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        ems["edited_pixel_values"] = edited_pixel_values

        target_parsing_values = torch.stack([example["target_parsing_values"] for example in examples])
        target_parsing_values = target_parsing_values.to(memory_format=torch.contiguous_format).float()
        ems["target_parsing_values"] = target_parsing_values

        input_ids = torch.stack([tokenize_captions(example["input_ids"]) for example in examples]).squeeze(1)
        ems["input_ids"] = input_ids

        if "reference_pixel_values" in examples[0]:
            reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
            reference_pixel_values = reference_pixel_values.to(memory_format=torch.contiguous_format).float()
            ems["reference_pixel_values"] = reference_pixel_values

        return ems

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader= torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        # collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encoder, vision_encoder and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vision_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("TailorEdit", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # losses = []
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the vision embedding for conditioning if reference image is passed.
                if "reference_pixel_values" in batch:
                    ref_image_embeds = vision_encoder(batch["reference_pixel_values"]).image_embeds
                    ref_image_embeds = ref_image_embeds.unsqueeze(1).to(device=latents.device)
                    added_cond_kwargs = ({"image_embeds": [ref_image_embeds]})
                else:
                    added_cond_kwargs = None

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                condition_image_embeds = batch["target_parsing_values"].to(weight_dtype).to(memory_format=torch.contiguous_format).float()

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds
                    condition_image_embeds = image_mask * condition_image_embeds

                    # Mask the visual info from ip-adapter
                    if added_cond_kwargs:
                        image_mask = image_mask.squeeze(-1)
                        ref_image_embeds = image_mask * ref_image_embeds
                        added_cond_kwargs = ({"image_embeds": [ref_image_embeds]})

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # ControlNet prepare
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=condition_image_embeds,
                    return_dict=False,
                )
                
                # Predict the noise residual and compute loss
                model_pred = unet(
                    concatenated_noisy_latents, 
                    timesteps, 
                    encoder_hidden_states, 
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample, 
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # if args.edit_localization:
                #     local_loss = get_edit_localization_loss(cross_attention_scores, batch["edit_mask_values"])
                #     loss = denoise_loss + args.edit_localization_weight * local_loss
                # else:
                #     loss = denoise_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(update_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # losses.append(train_loss)
                # draw_loss(losses, os.path.join(args.output_dir, "loss.jpg"))
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_state_dict = {}
                        for n, p in unet.named_parameters():
                            if args.lora_task in n:
                                save_state_dict[n] = p
                        torch.save(save_state_dict, f"{save_path}.pth")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                flag = True
                while flag:
                    flag = False
                    try:
                        pipeline = TailorEditPixPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=unwrap_model(unet),
                            controlnet=unwrap_model(controlnet),
                            scheduler=unwrap_model(noise_scheduler),
                            text_encoder=unwrap_model(text_encoder),
                            vae=unwrap_model(vae),
                            image_encoder=unwrap_model(vision_encoder),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                    except:
                        flag = True
                
                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    generator,
                    val_dataloader,
                    global_step
                )

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = TailorEditPixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            controlnet=unwrap_model(controlnet),
            image_encoder=unwrap_model(vision_encoder),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
                val_dataloader,
                global_step
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()