import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

im_path = "/home/ekagra/personal/projects/ComputerVision/data/turningtorso.jpg"
target_image = np.array(Image.open(im_path))

# display the target image
plt.imshow(target_image)
plt.title("Select 30 points")
plt.axis("on")

# Use ginput() to manually select 30 points
selected_points = plt.ginput(
    30, timeout=0
)  # timeout=0 mean infinite time to set the 30 points
plt.show()

# convert the selected points to a numpy array
selected_points = np.array(selected_points)

print(f"#Selected points: {len(selected_points)}\n")

# save the selected points
np.savetxt(
    "/home/ekagra/personal/projects/ComputerVision/data/turningtorso.txt",
    selected_points.astype(int),
)
