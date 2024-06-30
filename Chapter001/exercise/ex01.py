from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image
image_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
image = Image.open(image_path).convert("L")  # Load as grayscale


# Function to plot the image and its contours
def plot_contours(image, sigma_values):
    fig, axes = plt.subplots(1, len(sigma_values), figsize=(20, 5))

    for i, sigma in enumerate(sigma_values):
        # Apply Gaussian blur
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Convert to numpy array and find edges using numpy gradient
        blurred_array = np.array(blurred_image)
        grad_x, grad_y = np.gradient(blurred_array)
        edges = np.sqrt(grad_x**2 + grad_y**2)

        # Plot the blurred image
        axes[i].imshow(blurred_image, cmap="gray")
        axes[i].set_title(f"σ = {sigma}")
        axes[i].axis("off")

        # Overlay the contours
        contour_levels = [edges.max() * 0.5]  # Set a threshold for contour visibility
        axes[i].contour(edges, levels=contour_levels, colors="r")

    plt.tight_layout()
    plt.show()


# Define sigma values for increasing blur
sigma_values = [1, 5, 10, 20]

# Plot contours for each sigma value
plot_contours(image, sigma_values)
