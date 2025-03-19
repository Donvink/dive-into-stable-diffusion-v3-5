import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/stable-diffusion-3.5-medium",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Path to lora weights.",
    )
    parser.add_argument(
        "--aesthetic_predictor_name_or_path",
        type=str,
        default="models/aesthetics-predictor-v1-vit-large-patch14",
        help="Path to aesthetics predictor model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A pokemon with green eyes and red legs.",
        help="Path to lora weights.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height in pixels of the generated image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width in pixels of the generated image.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality image at the "
            "expense of slower inference.",
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help=(
            "Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, "
            "usually at the expense of lower image quality.",
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images that should be generated.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="datas/aesthetic_predictor_images/a-photo-of-an-astronaut-riding-a-horse.png",  # 6.54
        help="Path to image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/images/lora_sd3_debug",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Sanity checks
    if args.pretrained_model_name_or_path is None:
        if args.aesthetic_predictor_name_or_path is None:
            raise ValueError("Specify either `--pretrained_model_name_or_path` or `--aesthetic_predictor_name_or_path`")
        if args.image_path is None:
            raise ValueError("Specify `--image_path` without generating images")

    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrained_model_name_or_path is not None:
        # Load pretrained model
        pipe = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model_name_or_path)

        # Load lora weights
        if args.lora_weights_path is not None:
            pipe.load_lora_weights(args.lora_weights_path)

        pipe = pipe.to(device)

        # Generate images
        images = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.aesthetic_predictor_name_or_path is not None:
            # Load the aesthetics predictor
            aesthetics_predictor = AestheticsPredictorV1.from_pretrained(args.aesthetic_predictor_name_or_path).to(device).eval()
            clip_processor = CLIPProcessor.from_pretrained(args.aesthetic_predictor_name_or_path)

            # Preprocess the image
            images = [img.convert("RGB") for img in images]
            inputs = clip_processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            # Run inference
            with torch.no_grad():
                outputs = aesthetics_predictor(**inputs)
            predictions = outputs.logits.squeeze().tolist()  # 获取所有分数
            predictions = [predictions] if isinstance(predictions, float) else predictions

            # save image
            for i, (image, prediction) in enumerate(zip(images, predictions)):
                print(f"Aesthetics score: {args.prompt}_{i}.png, {prediction:.2f}")
                if args.output_dir is not None:
                    image.save(os.path.join(args.output_dir, f"{args.prompt}_{i}_{prediction:.2f}.png"))
        else:
            # save image
            if args.output_dir is not None:
                for i, image in enumerate(images):
                    image.save(os.path.join(args.output_dir, f"{args.prompt}_{i}.png"))
    else:
        # Load the aesthetics predictor
        aesthetics_predictor = AestheticsPredictorV1.from_pretrained(args.aesthetic_predictor_name_or_path).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(args.aesthetic_predictor_name_or_path)

        # Load sample image
        image = Image.open(args.image_path).convert("RGB")  # convert to RGB format

        # Preprocess the image
        inputs = clip_processor(
            images=image, 
            return_tensors="pt",
            padding=True
        ).to(device)

        # Inference for the image
        with torch.no_grad():
            outputs = aesthetics_predictor(**inputs)
        prediction = outputs.logits.item()  # get score

        print(f"{args.image_path}, Aesthetics score: {prediction:.2f}")


"""
Command lines:

python demo.py \
    --pretrained_model_name_or_path models/stable-diffusion-3.5-medium \
    --lora_weights_path outputs/lora_sd3_single_gpu \
    --aesthetic_predictor_name_or_path models/aesthetics-predictor-v1-vit-large-patch14 \
    --prompt "A pokemon with green eyes and red legs." \
    --height 1024 --width 1024 \
    --guidance_scale 4.5 \
    --max_sequence_length 77 \
    --num_images_per_prompt 4 \
    --output_dir outputs/images/sd3_lora_aesthetics

"""


if __name__ == "__main__":
    args = parse_args()
    main(args)



