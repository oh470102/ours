import cv2
import numpy as np
import os

# ! perspective warp with no interpolation
# def perspective_warp_no_interpolation(src_img, H, output_shape, tolerance=0.2):
#     # Initialize the output image with zeros
#     dst_img = np.zeros(output_shape, dtype=src_img.dtype)
    
#     # Get the height and width of the source and destination images
#     h, w = src_img.shape[:2]
#     h_dst, w_dst = output_shape[:2]

#     # Initialize a counter for zero-filled pixels
#     zero_filled_count = 0

#     # Iterate over every pixel in the source image
#     for y in range(h):
#         for x in range(w):
#             # Create a homogeneous coordinate for the source pixel
#             src_pt = np.array([x, y, 1])
            
#             # Apply the homography matrix
#             dst_pt = np.dot(H, src_pt)
            
#             # Normalize to get the (x, y) coordinates in the destination image
#             dst_pt = dst_pt / dst_pt[2]
#             dst_x, dst_y = dst_pt[:2]
            
#             # Round the coordinates to the nearest integer
#             int_dst_x = int(round(dst_x))
#             int_dst_y = int(round(dst_y))
            
#             # Check if the destination coordinates are within the image bounds and within tolerance
#             if (0 <= int_dst_x < w_dst and 0 <= int_dst_y < h_dst and 
#                 abs(dst_x - int_dst_x) <= tolerance and abs(dst_y - int_dst_y) <= tolerance):
                
#                 # Map the source pixel to the destination pixel
#                 dst_img[int_dst_y, int_dst_x] = src_img[y, x]
#             else:
#                 # Increment the zero-filled counter if coordinates are out of bounds or not within tolerance
#                 zero_filled_count += 1

#     # Print out how many of the original pixels got zero-filled
#     print(f"Number of zero-filled pixels: {zero_filled_count / (h * w) * 100}%")

#     return dst_img

# def process_video(input_path, output_path, H):
#     # Open the input video
#     cap = cv2.VideoCapture(input_path)
    
#     # Get the video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Apply the perspective warp with no interpolation
#         warped_frame = perspective_warp_no_interpolation(frame, H, (frame_height, frame_width, 3))
        
#         # Write the processed frame to the output video
#         out.write(warped_frame)
    
#     # Release everything if job is finished
#     cap.release()
#     out.release()

# # Example usage:
# input_video_path = './resources/retriever.mp4'
# output_video_path = './resources/retriever_warped.mp4'

# slant_factor = 0.1
# perspective_factor = 0.0002

# # Define source points (corners of the original image)
# dst_pts = [(0, 0), (319, 0), (319, 319), (0, 319)]
# src_pts = [(0, 0), (319, 80), (319, 319-80), (0, 319)]
# src_pts = np.float32(src_pts)
# dst_pts = np.float32(dst_pts)

# H, _ = cv2.findHomography(src_pts, dst_pts)
# print(H)

# process_video(input_video_path, output_video_path, H)
# !

def perspective_warp(image, src_pts, dst_pts):
    """
    Perform a perspective warp on a single image given source and destination points.

    :param image: Input image.
    :param src_pts: Source points for the perspective transformation.
    :param dst_pts: Destination points for the perspective transformation.
    :return: Warped image.
    """
    # Convert points to numpy arrays
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    
    # Compute the homography matrix
    h_matrix, _ = cv2.findHomography(src_pts, dst_pts)
    
    # Warp the image using the homography matrix
    warped_image = cv2.warpPerspective(image, h_matrix, (image.shape[1], image.shape[0]))
    
    return warped_image

def process_video(input_video_path, output_video_path, src_pts, dst_pts):
    """
    Process a video by applying a perspective warp to each frame and then rectifying it.

    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to save the output video file.
    :param src_pts: Source points for the perspective transformation.
    :param dst_pts: Destination points for the perspective transformation.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply perspective warp to the frame
        warped_frame = perspective_warp(frame, src_pts, dst_pts)
        
        # Rectify the warped frame to square shape
        # rectified_frame = inverse_perspective_warp(warped_frame, src_pts, dst_pts)
        
        # Resize rectified frame to 320x320 if necessary
        # rectified_frame = cv2.resize(warped_frame, (320, 320))
        
        # Write the processed frame to the output video
        out.write(warped_frame)
    
    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    input_video_path = './resources/retriever.mp4'
    output_video_path = './resources/retriever_warped.mp4'
    
    d = 90

    dst_pts = [(0, 0), (319, 0), (319, 319), (0, 319)]
    src_pts = [(0, d), (319, 0), (319, 319), (0, 319 - d)]

    
    process_video(input_video_path, output_video_path, src_pts, dst_pts)



# * VIEW FROM RIGHT
# dst_pts = [(0, 0), (319, 0), (319, 319), (0, 319)]
# src_pts = [(0, 0), (319, d), (319, 319-d), (0, 319)]

# * VIEW FROM LEFT
# dst_pts = [(0, 0), (319, 0), (319, 319), (0, 319)]
# src_pts = [(0, d), (319, 0), (319, 319), (0, 319 - d)]

# * VIEW FROM BOTTOM
# dst_pts = [(0, 0), (319, 0), (319, 319), (0, 319)]
# src_pts = [(0, 0), (319, 0), (319-d, 319-d), (d, 319 - d)]
