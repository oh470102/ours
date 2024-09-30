import os
import cv2

def process_videos(directory):
    # Get a list of all .mp4 files in the directory
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Create a folder for the current video
        video_folder = os.path.join(directory, os.path.splitext(video_file)[0])
        os.makedirs(video_folder, exist_ok=True)

        # Process frames
        frame_numbers = [1, 4, 8, 12, 16]
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if there are no more frames

            frame_count += 1
            if frame_count in frame_numbers:
                # Save the frame as a .png image
                frame_filename = os.path.join(video_folder, f'frame_{frame_count}.png')
                cv2.imwrite(frame_filename, frame)

        cap.release()

# Usage example
directories = ['../samples/Figure 1 - intro',
               '../samples/Figure 2 - position',
               '../samples/Figure 3 - size',
               '../samples/Figure 4 - perspective',]
for directory in directories:
    process_videos(directory)
