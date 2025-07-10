import argparse
import torch

from utils.utils import prepare_data
from src.model import load_pipeline, inference
from modules.ppn.ppn_model import PPN

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a testing script for FashionEdit.")
    parser.add_argument("--image_path", type=str, required=True, help="Path of the image, which you want to edit.")
    parser.add_argument("--parsing_path", type=str, required=True, help="Path of the parsing image, which you want to edit.")
    parser.add_argument("--instruction", type=str, required=True, help="Prompt about editing.")
    parser.add_argument("--reference_path", default=None, help="Path of the reference image.")
    parser.add_argument("--output_dir", type=str, default="results")
    
    parser.add_argument("--tailor_edit_path", type=str, default="./checkpoints/tailor_edit_model")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--pretrained_vision_model_name_or_path", type=str, default="h94/IP-Adapter")
    parser.add_argument("--parsing_prediction_network_path", type=str, default="./checkpoints/parsing_predict.pth")
    parser.add_argument("--ip_adapter_path",type=str, default="./checkpoints/ip-adapter_sd15.bin")
    parser.add_argument("--controlnet_path", type=str, default="./checkpoints/controlnet_model")
    parser.add_argument("--global_gate_path", type=str, default="./checkpoints/global_gate.pth")
    
    parser.add_argument("--seed", type=int, default=12, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True, action="store_true")
    parser.add_argument("--use_lora", default=True, help="Whether to use LORA for the textual cross attention layers.")
    parser.add_argument("--lora_alpha", type=float, default=1, help="LORA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LORA dropout parameter.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LORA rank parameter.")

    args = parser.parse_args()

    args.weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        args.weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        args.weight_dtype = torch.bfloat16

    return args

def main():
    args = parse_args()
    pipeline = load_pipeline(args, device)

    ppn = PPN(18)
    ppn.load_network(args.parsing_prediction_network_path)
    
    inference_data = prepare_data(args.image_path, args.parsing_path, args.instruction, args.reference_path)

    inference(pipeline, ppn, args, inference_data, device)

if __name__ == "__main__":
    main()