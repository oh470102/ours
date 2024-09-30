import os

# Directory where the videos are stored
video_dir = './resources/davis/'

# Directory where the .yaml files will be saved
yaml_dir = './configs/exp_config/'
if not os.path.exists(yaml_dir):
    os.makedirs(yaml_dir)

# Inversion and new prompt lists
inversion_prompts = [
  "Bear, walking, in the rocky enclosure.",
  "Swan, swimming, in the pond.",
  "Boat, sailing, in the open sea.",
  "Performer, breakdancing, in front of a house.",
  "Bus, driving, on the city street.",
  "Camel, standing, in the zoo enclosure.",
  "Car, driving, on the city street.",
  "Car, driving, around the roundabout.",
  "Car, driving, on the mountain road.",
  "Cow, grazing, in the grassy field.",
  "Dog, sniffing, in the yard.",
  "Dog, running, in the obstacle course.",
  "Car, drifting, in the stadium.",
  "Elephant, walking, in the zoo enclosure.",
  "Flamingo, standing, in the pond.",
  "Goat, walking, on the mountain cliff.",
  "Hiker, walking, in the rocky mountains.",
  "Athlete, playing hockey, in the park.",
  "Kid, playing football, in the field.",
  "Man, kite surfing, in the ocean.",
  "Dog, running, in the garden.",
  "Man, skateboarding, in the forest.",
  "Woman, walking, in the park.",
  "Duck, moving, by the riverbank.",
  "Duck, swimming, in the pond.",
  "Man, riding a motorcycle, on the roads.",
  "Man, riding a motorcycle, on a road.",
  "Man, doing parkour, in the urban park.",
  "Rhino, walking, in the zoo enclosure.",
  "Man, riding a scooter, on the urban road.",
  "Man, riding a snowboard, in the snowy mountains.",
  "Soccer ball, rolling, in the backyard.",
  "Woman, pushing a stroller, in the city.",
  "Windsurfer, sailing, in the ocean.",
  "Girl, swinging, in the playground.",
  "Athlete, playing tennis, on the tennis court.",
  "Toy train, moving, on the train track display."
]

new_prompts = [
  "Cat, walking, in the forest enclosure. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Duck, swimming, in the lake. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Sailboat, sailing, in the harbor. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, breakdancing, in front of a restaurant. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Van, driving, on the roads. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Giraffe, standing, in the safari enclosure. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ferrari, driving, in New York City. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ferrari, driving, around the countryroad. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ferrari, driving, along the seashore. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Sheep, grazing, in the pasture. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Cat, sniffing, in the garden. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Cat, running, in the park track. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ferrari, drifting, in the arena. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Horse, walking, in the safari enclosure. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Crane, standing, in the river. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Dog, walking, on the rocky cliff. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ironman, walking, on Mars. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, playing hockey, in the park. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Batman, playing football, outdoors. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ironman, kite surfing, in the river. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Cat, running, in the backyard. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, skateboarding, in the snow. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Darth-vader, walking, in the garden. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Chicken, moving, by the lake. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Goose, swimming, in the river. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, riding a motorcycle, in the city streets. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ironman, riding a motorcycle, in the city. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, doing parkour, in the city. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Elephant, walking, in the forest. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Batman, riding a scooter, on the city streets. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Spiderman, riding a snowboard, in the mountains. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Basketball, rolling, in the park. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Wonder Woman, pushing a stroller, in the park. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Sailboat, sailing, in the river. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ironman, swinging, in the playground. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Ironman, playing tennis, on the tennis court. 8k, high detailed, best quality, film grain, Fujifilm XT3",
  "Train, moving, on the rails. 8k, high detailed, best quality, film grain, Fujifilm XT3"
]

# Loop through each video in the directory and generate the .yaml file
for idx, video_file in enumerate(sorted(os.listdir(video_dir))):
    if video_file.endswith(".mp4"):
        video_name = os.path.splitext(video_file)[0]
        yaml_content = f"""# motion module v3_sd15

  motion_module:    "models/Motion_Module/v3_sd15_mm.ckpt"
  dreambooth_path: "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"

  W: 320
  H: 320
  L: 16

  model_config: "configs/model_config/model_config.yaml"
  cfg_scale: 7.5 # in default realistic classifer-free guidance
  negative_prompt: "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers"

  num_inference_step: 500 # the denosing step for both inversion and inference
  guidance_step: 150 # the step for guidance in inference
  warm_up_step: 15 # the warm up step for guidance in inference
  cool_up_step: 0 # the cool up step for guidance in inference
  grad_guidance_scale: 1.0 # Gradient guidance scale, it will be multiplied with the weight in each type of guidance; 
  grad_guidance_threshold: null # Gradient guidance threshold,number or null

  temp_guidance:
    weight_each: [20000, 10000]  # [20000, 10000, 8000]
    blocks:
      ['up_blocks.1', 'up_blocks.2'] # the length of "blocks" must match the length of "weight_each"
  app_guidance:
    weight_each: [0, 0] # 400, 200
    block_type: ["temp"] # determine the key in which block is used for appearance loss
    blocks:
      [ 'up_blocks.1','up_blocks.2'] # the length of "blocks" must match the length of "weight_each"

    cross_attn_blocks:
      [ 'up_blocks.1','up_blocks.2'] # the mask is only extracted from the "up_blocks.1", which is interpolated for other blocks.
    cross_attn_mask_tr_example: 0.3 # threshold for mask extraction of example video 
    cross_attn_mask_tr_app: 0.3  # threshold for mask extraction of appearance video

  video_path: 'resources/davis/{video_file}'
  inversion_prompt: "{inversion_prompts[idx]}"
  new_prompt: "{new_prompts[idx]}"
  obj_pairs: ["{inversion_prompts[idx].split(',')[0]}", "{new_prompts[idx].split(',')[0]}"]
  is_downsized: False
  resize_factor: 1.0
  use_motionclone: False
  use_spatial_module: False
  attention_component: 'v'
  """

    # Save the .yaml file
    yaml_path = os.path.join(yaml_dir, f"{video_name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}")
