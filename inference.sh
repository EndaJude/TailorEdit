export CUDA_VISIBLE_DEVICES=0

python inference.py \
    --image_path "./examples/1.add the scarf/image.jpg" \
    --parsing_path "./examples/1.add the scarf/parsing.png" \
    --instruction_path "add the scarf." \
    --reference_path "./examples/1.add the scarf/reference.jpg" \
    --ouput_dir "results" \
    --seed 12 \
    --resolution 512 \
    --enable_xformers_memory_efficient_attention