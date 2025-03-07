import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
import torch, argparse
from modify_model.modify_flux import set_spas_sage_attn_flux
import os, gc
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)

file_path = "evaluate/datasets/video/prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Flux Evaluation")

    parser.add_argument("--use_spas_sage_attn", action="store_true", help="Use Sage Attention")
    parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/image/flux_sparge",
        help="out_path",
    )
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/flux_saved_state_dict.pt",
        help="model_out_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(file_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()

    model_id = "black-forest-labs/FLUX.1-dev"
    if args.tune == True:
        os.environ["TUNE_MODE"] = "1"  # enable tune mode

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose)

        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        for i, prompt in enumerate(prompts[:10]):
            image = pipe(
                prompt.strip(),
                height=1024,  # tune in 512 and infer in 1024 result in a good performance
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
            ).images[0]

            del image
            gc.collect()
            torch.cuda.empty_cache()

        saved_state_dict = extract_sparse_attention_state_dict(transformer)
        torch.save(saved_state_dict, args.model_out_path)

    else:
        os.environ["TUNE_MODE"] = ""  # disable tune mode

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            local_files_only=True,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose)
            saved_state_dict = torch.load(args.model_out_path)
            load_sparse_attention_state_dict(transformer, saved_state_dict)

        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        for i, prompt in enumerate(prompts):
            image = pipe(
                prompt.strip(),
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
            ).images[0]

            image.save(f"{args.out_path}/{i}.jpg")
            del image
            gc.collect()
            torch.cuda.empty_cache()
