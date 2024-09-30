


# fox --> dog
python invert.py --config configs/inference_config/fox2.yaml &&
python sample.py --config configs/inference_config/fox2.yaml --gen_prompt "Dog, trotting, in the park, 8k, high detailed, best quality, film grain, Fujifilm XT3" &&
python sample.py --config configs/inference_config/fox2.yaml --gen_prompt "Dog, trotting, in the park, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "down" &&
python sample.py --config configs/inference_config/fox2.yaml --gen_prompt "Dog, trotting, in the park, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "resize" --resize_factor 0.6 &&
python sample.py --config configs/inference_config/fox2.yaml --gen_prompt "Dog, trotting, in the park, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "resize" --resize_factor 1.7 &&





# Tiger --> Cat
python invert.py --config configs/inference_config/tiger.yaml &&
python sample.py --config configs/inference_config/tiger.yaml --gen_prompt "Cat, walking, on grass, 8k, high detailed, best quality, film grain, Fujifilm XT3"  &&
python sample.py --config configs/inference_config/tiger.yaml --gen_prompt "Cat, walking, on grass, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "shift" --shift_dir "right" &&
python sample.py --config configs/inference_config/tiger.yaml --gen_prompt "Cat, walking, on grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "resize" --resize_factor 0.6 &&
python sample.py --config configs/inference_config/tiger.yaml --gen_prompt "Cat, walking, on grass, 8k, high detailed, best quality, film grain, Fujifilm XT3"  --transformation "resize" --resize_factor 1.7 

