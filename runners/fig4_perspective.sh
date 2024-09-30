


# retriever --> lion
python invert.py --config configs/inference_config/retriever.yaml &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3"  &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "left" --resize_factor 1.0 &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "left" --resize_factor 2.0 &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "left" --resize_factor 3.0 &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "right" --resize_factor 1.0 &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "right" --resize_factor 2.0 &&
python sample.py --config configs/inference_config/retriever.yaml --gen_prompt "Lion, moves its head, in the wild, 8k, high detailed, best quality, film grain, Fujifilm XT3" --transformation "warp" --shift_dir "right" --resize_factor 3.0