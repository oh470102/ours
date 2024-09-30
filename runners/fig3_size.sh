


# wolf --> fox
python invert.py --config configs/inference_config/wolf.yaml &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "resize" --resize_factor 1.25 &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 1.5 &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 1.75 &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.80 &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.60 &&
python sample.py --config configs/inference_config/wolf.yaml --gen_prompt "Fox, moves its head, in the forest, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.40 

# bear --> tiger
python invert.py --config configs/inference_config/bear.yaml &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "resize" --resize_factor 1.3 &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 1.7 &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 2.0 &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.80 &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.60 &&
python sample.py --config configs/inference_config/bear.yaml --gen_prompt "Tiger, walking, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 0.40 

