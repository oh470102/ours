
import cv2
import os

# Path to the folder containing subfolders with frames
input_folder = '/home/yeon/Downloads/DAVIS/JPEGImages/480p'

# Path to save the output videos
output_folder = './resources/davis/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to convert frames to video
def frames_to_video(input_path, output_path, fps=16):
    # Get all the .jpg files from the directory
    images = [img for img in os.listdir(input_path) if img.endswith(".jpg")]
    images.sort()  # Sort by frame number
    
    if not images:
        print(f"No images found in {input_path}")
        return
    
    # Get the size of the images
    frame = cv2.imread(os.path.join(input_path, images[0]))
    height, width, layers = frame.shape
    
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for image in images:
        img_path = os.path.join(input_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    # Release the video writer object
    video.release()

# Loop through each subfolder in the '480p' folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)
    
    if os.path.isdir(folder_path):  # Check if it's a directory
        # Set the output video file name
        output_video_path = os.path.join(output_folder, f'{folder_name}.mp4')
        
        # Convert frames to video
        print(f"Converting frames in '{folder_name}' to video...")
        frames_to_video(folder_path, output_video_path)
        print(f"Saved video: {output_video_path}")
