#!/bin/bash

# Loop over each YAML file in the configs/exp_config directory
for config_file in ./configs/exp_config/q/*.yaml; do
  # Extract the base name of the file (without the directory and extension)
  base_name=$(basename "$config_file" .yaml)
  
  echo "Processing $base_name..."

  # Run the inversion command
  # python invert.py --config "$config_file" --num_inference_steps 1000 &&

  # Run the sample command with default gen_prompt
  python sample.py --config "$config_file" --save_dir "./samples/quant_exp/ours_qkv_combination/ours_q" --num_inference_steps 500 && 

  # Run the sample commands with various shift directions
#   python sample.py --config "$config_file" --gen_prompt "$base_name, moving, in the scene, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "up" &&
#   python sample.py --config "$config_file" --gen_prompt "$base_name, moving, in the scene, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "left" &&
#   python sample.py --config "$config_file" --gen_prompt "$base_name, moving, in the scene, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "right" &&
#   python sample.py --config "$config_file" --gen_prompt "$base_name, moving, in the scene, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "down"

  echo "Completed processing $base_name."
done
