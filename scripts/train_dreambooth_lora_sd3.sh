export MODEL_NAME="models/stable-diffusion-3.5-medium"
export INSTANCE_DIR="datas/dogs"
export OUTPUT_DIR="outputs/train_dreambooth_lora_sd3"

accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --instance_data_dir=${INSTANCE_DIR} \
  --output_dir=${OUTPUT_DIR} \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --weighting_scheme="logit_normal" \
  --seed="0" \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25
