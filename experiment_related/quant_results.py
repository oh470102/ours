import os
import torch
import clip
import cv2
from PIL import Image
from itertools import combinations

# Load the CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames_from_video(video_path, frame_rate=1):
    """
    Extract frames from the video at a given frame rate (frames per second).
    
    Args:
    video_path (str): Path to the .mp4 video file.
    frame_rate (int): Number of frames to extract per second. Default is 1.
    
    Returns:
    List of PIL.Images: Extracted frames from the video.
    """
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    interval = max(1, int(fps / frame_rate))  # Extract one frame per frame_rate seconds
    
    frames = []
    count = 0
    success = True
    
    while success:
        success, frame = video.read()
        if success and count % interval == 0:
            # Convert frame (which is in BGR format) to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1

    video.release()
    return frames

def compute_frame_text_similarity(frames, text):
    """
    Computes the average CLIP cosine similarity between all video frames and text.
    
    Args:
    frames (list of PIL.Images): List of video frames.
    text (str): Text description for comparison.

    Returns:
    float: Average cosine similarity between frames and text.
    """
    # Preprocess the frames and text
    frame_tensors = torch.stack([preprocess(frame).to(device) for frame in frames])
    text_tokens = clip.tokenize([text]).to(device)

    # Encode frames and text using CLIP model
    with torch.no_grad():
        frame_embeddings = model.encode_image(frame_tensors)
        text_embedding = model.encode_text(text_tokens)

    # Normalize the embeddings
    frame_embeddings /= frame_embeddings.norm(dim=-1, keepdim=True)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    # Compute cosine similarities between each frame and the text
    similarities = (frame_embeddings @ text_embedding.T).squeeze()

    # Return the average similarity
    return similarities.mean().item()

def compute_temporal_consistency(frames):
    """
    Computes the average CLIP similarity between all possible pairs of video frames
    to measure temporal consistency (smoothness).
    
    Args:
    frames (list of PIL.Images): List of video frames.
    
    Returns:
    float: Average cosine similarity between all possible pairs of frames.
    """
    # Preprocess the frames
    frame_tensors = torch.stack([preprocess(frame).to(device) for frame in frames])

    # Encode frames using CLIP model
    with torch.no_grad():
        frame_embeddings = model.encode_image(frame_tensors)

    # Normalize the embeddings
    frame_embeddings /= frame_embeddings.norm(dim=-1, keepdim=True)

    # Compute cosine similarities between all pairs of frames
    similarities = []
    for i, j in combinations(range(len(frame_embeddings)), 2):
        similarity = (frame_embeddings[i] @ frame_embeddings[j].T).item()
        similarities.append(similarity)

    # Return the average similarity
    return sum(similarities) / len(similarities) if similarities else 0.0

def process_videos_in_directory(directory, frame_rate=1):
    """
    Processes all .mp4 videos in the directory, computing similarity and consistency
    for each video and returning the average scores.
    
    Args:
    directory (str): Path to the directory containing .mp4 video files.
    description (str): Text description to compare the frames to.
    frame_rate (int): Number of frames to extract per second. Default is 1.
    
    Returns:
    tuple: Average frame-text similarity and temporal consistency for all videos.
    """
    video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    total_similarity = 0.0
    total_consistency = 0.0
    num_videos = len(video_files)
    
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        frames = extract_frames_from_video(video_path, frame_rate=frame_rate)
        

        description = video_file.replace("_", " ").replace(",", "")
        description = description.split("8k")[0].strip()

        similarity = compute_frame_text_similarity(frames, description)
        # print(description)
        consistency = compute_temporal_consistency(frames)
        
        total_similarity += similarity
        total_consistency += consistency
        # print(f"Processed {video_file}: Similarity={similarity}, Consistency={consistency}")
    
    avg_similarity = total_similarity / num_videos if num_videos > 0 else 0.0
    avg_consistency = total_consistency / num_videos if num_videos > 0 else 0.0
    
    return avg_similarity, avg_consistency

# Example usage
directory_path = "./samples/quant_exp/ours_new"
frame_rate = 8  # Extract 8 frames per second

# Process all videos in the directory and compute the average similarity and consistency
avg_similarity, avg_consistency = process_videos_in_directory(directory_path, frame_rate=frame_rate)

print(f"Average frame-text similarity across videos: {avg_similarity}")
print(f"Average temporal consistency across videos: {avg_consistency}")
