import sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
CATVTON_ROOT = BASE_DIR / "virtual_try_on" / "CatVTON_edited"
sys.path.insert(0, str(CATVTON_ROOT))
import argparse
import os
from datetime import datetime
import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from PIL import Image

def make_gradio_like_person_dict(person_path: str):
    bg = Image.open(person_path).convert("RGBA")
    blank = Image.new("RGBA", bg.size, (0, 0, 0, 0))  # fully transparent
    return {"background": person_path, "layers": [blank]}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=("The Path to the checkpoint of trained tryon model."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint",
        action="store_true",
        help="Whether to repaint the result image with the original background.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# args = parse_args()
args = argparse.Namespace(
    base_model_path="booksforcharlie/stable-diffusion-inpainting",
    resume_path="zhengchong/CatVTON",
    output_dir="resource/demo/output",
    width=768,
    height=1024,
    repaint=False,
    allow_tf32=True,
    mixed_precision="fp16",
)

repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device="cuda",
)
# AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)


def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type,
):
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = mask.convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)
    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(person_image, cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )

    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid(
        [person_image, masked_person, cloth_image, result_image], 1, 4
    )
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image


def run_tryon(
    person_path,
    cloth_path,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type,
):

    person_image = make_gradio_like_person_dict(person_path)

    return submit_function(
        person_image,
        cloth_path,
        cloth_type,
        num_inference_steps,
        guidance_scale,
        seed,
        show_type,
    )


print("********CatVTON service is ready.********")
