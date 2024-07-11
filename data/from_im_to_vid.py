import cv2
import os

image_folder = '/home/ekagra/personal/projects/ComputerVision/data/jkfaces'
video_name = '/home/ekagra/personal/projects/ComputerVision/data/jkfaces/jkfaces.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Ensure images are in the correct order

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
