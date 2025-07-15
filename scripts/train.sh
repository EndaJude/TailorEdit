export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_ID="InstructFashion"
export CUDA_VISIBLE_DEVICES=0

accelerate launch --mixed_precision="fp16" ./scripts/train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --output_dir="output" \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 \
    --train_batch_size=4 \
    --num_train_epochs=20 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=10 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \