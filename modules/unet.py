from contextlib import nullcontext
import torch.nn.functional as F
from torch import nn
from diffusers import UNet2DConditionModel
from diffusers.utils import is_accelerate_available, logging, is_torch_version
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.models.embeddings import MultiIPAdapterImageProjection
from .attention_processor import MoLoAttnProcessor, MoLoAttnProcessor2_0
from peft.tuners.lora.layer import Linear as LoraLinear
from typing import Any, Dict
import torch
import types

logger = logging.get_logger(__name__)

def init_moe_in_lora(unet: UNet2DConditionModel):

    def forward_moe(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            assert len(self.active_adapters) == 1, "The number of actived adapters more than 1."
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    B, L, D = result.shape
                    lora_output = lora_B(lora_A(dropout(x))) * scaling
                    stacked_output = torch.stack([result, lora_output], dim=-1)
                    moe_weight = self.moe_gate(stacked_output.permute((0, 3, 1, 2)).reshape(B, -1))
                    moe_weight = F.softmax(moe_weight, dim=-1).unsqueeze(1).unsqueeze(2)
                    result = (stacked_output * moe_weight).sum(dim=-1)
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)

        return result

    channel_dict = {
        "down_blocks.0": {"sequence_length": 4096,   "dimension": 320},
        "down_blocks.1": {"sequence_length": 1024,   "dimension": 640},
        "down_blocks.2": {"sequence_length": 256,    "dimension": 1280},
        "mid_block":     {"sequence_length": 64,     "dimension": 1280},
        "up_blocks.1":   {"sequence_length": 256,    "dimension": 1280},
        "up_blocks.2":   {"sequence_length": 1024,   "dimension": 640},
        "up_blocks.3":   {"sequence_length": 4096,   "dimension": 320},
    }
    for n, m in unet.named_modules():
        if not any(kw in n for kw in ['to_k', 'to_v', 'to_q']):
            continue
        if isinstance(m, LoraLinear):
            num_experts = 2
            seq_len, dim = channel_dict[n.split('.attentions')[0]].values()
            seq_len = 77 if "attn2.to_k" in n or "attn2.to_v" in n else seq_len
            m.moe_gate = torch.nn.Linear(seq_len * dim * num_experts, num_experts, bias=False)
            m.forward = types.MethodType(forward_moe, m)

def set_unet_weight_requires_grad_(unet: UNet2DConditionModel):
    for n, p in unet.named_parameters():
        if "to_k_ip" in n or "to_v_ip" in n:
            p.requires_grad_(True)
        p.requires_grad_(True)

def set_unet_weight_requires_grad(unet: UNet2DConditionModel):
    for n, p in unet.named_parameters():
        if "moe_gate" in n:
            p.requires_grad_(True)
        if any(keyword in n for keyword in ["attn1.to_k", "attn1.to_v", "attn1.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_q"]):
            continue
        p.requires_grad_(True)

def convert_ip_adapter_attn_to_diffusers(unet, state_dicts, low_cpu_mem_usage=False):

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = unet.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()

            else:
                # attn_processor_class = (
                #     MoLoAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else MoLoAttnProcessor
                # )
                attn_processor_class = MoLoAttnProcessor
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

                key_id += 2

        return attn_procs

def load_ip_adapter_weights(unet, state_dicts, low_cpu_mem_usage=False):
    if not isinstance(state_dicts, list):
        state_dicts = [state_dicts]

    # Kolors Unet already has a `encoder_hid_proj`
    if (
        unet.encoder_hid_proj is not None
        and unet.config.encoder_hid_dim_type == "text_proj"
        and not hasattr(unet, "text_encoder_hid_proj")
    ):
        unet.text_encoder_hid_proj = unet.encoder_hid_proj

    # Set encoder_hid_proj after loading ip_adapter weights,
    # because `IPAdapterPlusImageProjection` also has `attn_processors`.
    unet.encoder_hid_proj = None

    attn_procs = convert_ip_adapter_attn_to_diffusers(unet, state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
    unet.set_attn_processor(attn_procs)

    # convert IP-Adapter Image Projection layers to diffusers
    image_projection_layers = []
    for state_dict in state_dicts:
        image_projection_layer = unet._convert_ip_adapter_image_proj_to_diffusers(
            state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
        )
        image_projection_layers.append(image_projection_layer)

    unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
    unet.config.encoder_hid_dim_type = "ip_image_proj"

    unet.to(dtype=unet.dtype, device=unet.device)

def process_encoder_hidden_states_new(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if added_cond_kwargs:
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )

                if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                    encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.encoder_hid_proj(image_embeds)
            else:
                image_embeds = [None]
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
        return encoder_hidden_states

def unet_store_cross_attention_scores(unet, attention_scores, layers=7):
    from diffusers.models.attention_processor import Attention

    UNET_LAYER_NAMES = [
        # "down_blocks.0",
        # "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        # "up_blocks.2",
        # "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            if attention_probs.shape[-1] == 77:
                attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, MoLoAttnProcessor2_0):
                hs = module.processor.hidden_size
                module.set_processor(MoLoAttnProcessor(hs))
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet

def get_edit_localization_loss_each_layer(cross_attention_scores, masks, loss_fn=F.l1_loss):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, _, _ = masks.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    masks = F.interpolate(
        masks.unsqueeze(1), size=(size, size), mode="bilinear", antialias=True
    ).view(b, 1, -1)  # (b, 1, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )

    masks = masks.unsqueeze(-1).repeat(1, num_heads, 1, num_text_tokens)

    loss = loss_fn(cross_attention_scores, masks)

    return loss

def get_edit_localization_loss(cross_attention_scores, edit_mask):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        loss += get_edit_localization_loss_each_layer(v, edit_mask)
    
    return loss / num_layers 