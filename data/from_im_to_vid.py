import cv2
import os

# Set the path to the folder containing the images
image_folder = (
    "/home/ekagra/personal/projects/ComputerVision/data/jkfaces"  # Update this path
)
video_name = "/home/ekagra/personal/projects/ComputerVision/data/jkfaces/jkfaces_orig.avi"  # Name of the output video file

# Get the list of images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Ensure the images are sorted in the correct order

# Read the first image to get the size (assuming all images are the same size)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the video codec and create VideoWriter object
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"DIVX"), 1, (width, height))

# Loop through images and write them to the video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release the video writer
cv2.destroyAllWindows()
video.release()

print(f"Video saved as {video_name}")
