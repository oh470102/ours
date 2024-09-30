


# dog --> wolf
python invert.py --config configs/inference_config/dog.yaml &&
python sample.py --config configs/inference_config/dog.yaml --gen_prompt "Wolf, moving, on the grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" &&
python sample.py --config configs/inference_config/dog.yaml --gen_prompt "Wolf, moving, on the grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "up" &&
python sample.py --config configs/inference_config/dog.yaml --gen_prompt "Wolf, moving, on the grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "left" &&
python sample.py --config configs/inference_config/dog.yaml --gen_prompt "Wolf, moving, on the grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "right" &&
python sample.py --config configs/inference_config/dog.yaml --gen_prompt "Wolf, moving, on the grass, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "down" 


# fox --> parrot
python invert.py --config configs/inference_config/fox.yaml &&
python sample.py --config configs/inference_config/fox.yaml --gen_prompt "Parrot, moves its head, in the jungle, 8k, high detailed, best quality, film grain, Fujifilm XT3" &&
python sample.py --config configs/inference_config/fox.yaml --gen_prompt "Parrot, moves its head, in the jungle, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "up" &&
python sample.py --config configs/inference_config/fox.yaml --gen_prompt "Parrot, moves its head, in the jungle, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "left" &&
python sample.py --config configs/inference_config/fox.yaml --gen_prompt "Parrot, moves its head, in the jungle, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "right" &&
python sample.py --config configs/inference_config/fox.yaml --gen_prompt "Parrot, moves its head, in the jungle, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "shift" --shift_dir "down" 

