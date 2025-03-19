export MODEL_NAME="models/stable-diffusion-3.5-medium"
export AESTHETIC_MODEL="models/improved-aesthetic-predictor"
export AESTHETIC_MODEL_NAME="ava+logos-l14-linearMSE.pth"
export CLIP_MODEL="models/clip-vit-large-patch14"
export OUTPUT_DIR="outputs/train_aesthetic_ddpo_sd3"

accelerate launch --num_processes=1 --num_machines=1 train_aesthetic_ddpo_sd3.py \
    --pretrained_model=${MODEL_NAME} \
    --aesthetic_model_path=${AESTHETIC_MODEL} \
    --aesthetic_model_filename=${AESTHETIC_MODEL_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --rank=4 \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --seed=1337
