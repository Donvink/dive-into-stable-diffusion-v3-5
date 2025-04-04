export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"
export OUTPUT_DIR="outputs/train_dpo_sd_v1_5"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16"  train_dpo_sd_v1_5.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir=$OUTPUT_DIR
