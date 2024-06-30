import numpy as np
from PIL import Image


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in eq (11) A. Chambolle (2005).
    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, steplength, tolerance for stop criterion.
    Output: denoised and detextured image, texture residual."""

    m, n = im.shape  # size of noisy image

    # Initialize
    u = U_init
    Px = im  # x-component of the dual field
    Py = im  # y-component of the dual field
    error = 1

    while error > tolerance:
        U_old = u

        # gradient of primal variable
        grad_Ux = np.roll(u, shift=-1, axis=1) - u  # x-component of U's gradient
        grad_Uy = np.roll(u, shift=-1, axis=0) - u  # y-component of Y's gradient

        # update the dual variable
        new_Px = Px + (tau / tv_weight) * grad_Ux
        new_Py = Py + (tau / tv_weight) * grad_Uy
        new_norm = np.maximum(1, np.sqrt(new_Px**2 + new_Py**2))

        Px = new_Px / new_norm  # update the x-component (dual)
        Py = new_Py / new_norm  # update the y-component (dual)

        # update the primal variable
        RxPx = np.roll(Px, 1, axis=1)  # right x-translation of x-component
        RyPy = np.roll(Py, 1, axis=0)  # right y-translation of y-component

        DivP = (Px - RxPx) + (Py - RyPy)  # divergence of the dual field
        u = im + tv_weight * DivP

        # update the error
        error = np.linalg.norm(u - U_old) / np.sqrt(n * m)
        print(error)
    return u, im - u  # denoised image and texture residual


if __name__ == "__main__":
    im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    im = np.array(Image.open(im_path).convert("L"))
    U, T = denoise(im=im, U_init=im)

    pil_U = Image.fromarray(U)
    pil_U.show()
