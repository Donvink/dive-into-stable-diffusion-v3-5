export MODEL_NAME="models/stable-diffusion-3.5-medium"
export LORA_MODEL_NAME="outputs/ddpo"
export AP_MODEL_NAME="models/aesthetics-predictor-v1-vit-large-patch14"
export OUTPUT_DIR="outputs/images/demo"

python demo.py \
    --pretrained_model_name_or_path=${MODEL_NAME} \
    --lora_weights_path=${LORA_MODEL_NAME} \
    --aesthetic_predictor_name_or_path=${AP_MODEL_NAME} \
    --prompt "a photo of an astronaut riding a horse." \
    --height 1024 --width 1024 \
    --guidance_scale 4.5 \
    --max_sequence_length 77 \
    --num_images_per_prompt 4 \
    --num_inference_steps 40 \
    --output_dir=${OUTPUT_DIR}


