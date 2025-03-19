# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python examples/train_aesthetic_ddpo_sd3.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
import sys

# # 获取当前文件的绝对路径，并向上导航到父目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# src_dir = os.path.join(parent_dir, 'src')

# # 将src目录添加到Python的搜索路径
# sys.path.append(src_dir)

from dataclasses import dataclass, field
import numpy as np
from transformers import HfArgumentParser
from trl import DDPOConfig
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from src.rewards import aesthetic_scorer
from src.ddpo_sd3_pipeline import DDPOStableDiffusion3Pipeline
from src.ddpo_sd3_trainer import DDPOStableDiffusion3Trainer


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"models/stable-diffusion-3.5-medium"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion-3.5"`):
            HuggingFace repo to save model weights to.
        aesthetic_model_path (`str`, *optional*, defaults to `"models/improved-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        aesthetic_model_filename (`str`, *optional*, defaults to `"ava+logos-l14-linearMSE.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        clip_model_path (`str`, *optional*, defaults to `"models/clip-vit-large-patch14"`):
            Hugging Face model ID for aesthetic scorer model weights.
        rank (`int`, *optional*, defaults to `4`):
            The dimension of the LoRA update matrices.
        lora_layers (`str`, *optional*, defaults to `None`):
            The transformer block layers to apply LoRA training on.
        lora_blocks (`str`, *optional*, defaults to `None`):
            The transformer blocks to apply LoRA training on.
        
    """

    pretrained_model: str = field(
        default="models/stable-diffusion-3.5-medium", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion-3.5", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    aesthetic_model_path: str = field(
        default="models/improved-aesthetic-predictor",
        metadata={"help": "Local path or Hugging Face model ID for aesthetic scorer model weights."},
    )
    aesthetic_model_filename: str = field(
        default="ava+logos-l14-linearMSE.pth",
        metadata={"help": "Local or Hugging Face model filename for aesthetic scorer model weights."},
    )
    clip_model_path: str = field(
        default="models/clip-vit-large-patch14",
        metadata={"help": "Local path or Hugging Face model ID for clip model."},
    )

    rank: int = field(default=4, metadata={"help": "The dimension of the LoRA update matrices."})
    lora_layers: str = field(
        default=None,
        metadata={"help": "The transformer block layers to apply LoRA training on. Please specify the layers in a comma seperated string."},
    )
    lora_blocks: str = field(
        default=None,
        metadata={"help": "The transformer blocks to apply LoRA training on. Please specify the block numbers in a comma seperated manner."},
    )

    output_dir: str = field(
        default="outputs/ddpo",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./outputs",
    }

    if script_args.lora_layers is not None:
        target_modules = [layer.strip() for layer in script_args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    if script_args.lora_blocks is not None:
        target_blocks = [int(block.strip()) for block in script_args.lora_blocks.split(",")]
        target_modules = [
            f"transformer_blocks.{block}.{module}" for block in target_blocks for module in target_modules
        ]
    transformer_lora_config = LoraConfig(
        r=script_args.rank,
        lora_alpha=script_args.rank,
        init_lora_weights="gaussian",
        # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        target_modules=target_modules,
    )

    pipeline = DDPOStableDiffusion3Pipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        lora_config=transformer_lora_config,
    )

    aesthetic_predictor = aesthetic_scorer(
        script_args.aesthetic_model_path,
        script_args.aesthetic_model_filename,
        script_args.clip_model_path,
    )

    trainer = DDPOStableDiffusion3Trainer(
        training_args,
        aesthetic_predictor,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(script_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
