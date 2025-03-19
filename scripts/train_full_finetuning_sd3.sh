export MODEL_NAME="models/stable-diffusion-3.5-medium"
export TRAIN_DATA_DIR="datas/pokemon"
export OUTPUT_DIR="outputs/train_full_finetuning_sd3"

accelerate launch --num_processes=1 --num_machines=1 train_full_finetuning_sd3.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --dataset_name=${TRAIN_DATA_DIR} \
  --output_dir=${OUTPUT_DIR} \
  --mixed_precision="fp16" \
  --resolution=1024\
  --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=1 \
  --checkpointing_steps=1000 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant"\
  --lr_warmup_steps=0 \
  --weighting_scheme="logit_normal" \
  --seed=1337 \
  --validation_prompt="A pokemon with green eyes and red legs." \
  --num_validation_images=4

