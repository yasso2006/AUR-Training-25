# Setup Commands: (inside VSCode terminal)
## (one-time) python -m venv .venv
## (Windows: every re-open) ./.venv/Scripts/activate.bat
## (Other systems: every re-open) ./.venv/Scripts/activate
## (one-time) pip install matplotlib opencv-python numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2


def convolve(image, kernel):

    """
    Apply a convolution to an image using a given kernel.

    Your code should handle different kernel sizes - not necessarily 3x3 kernels
    """
    if (kernel.shape[0] % 2 == 0) or (kernel.shape[1] % 2 == 0):
        print("unvalid kernel input")
        return False
    else:
        flipped = np.flipud(np.fliplr(kernel))
        k_h, k_w = flipped.shape
        p_h, p_w = k_h//2, k_w//2

        padded = np.pad(image, ((p_h, p_h), (p_w, p_w)), mode='constant', constant_values=0)
        output = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + k_h, j:j + k_w]
                output[i, j] = np.sum(region * flipped)
        return output

    


# Take notice that OpenCV handles the image as a numpy array when opening it
img = cv2.imread('messi.jpeg', cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

median = cv2.medianBlur(img, 5)
axes[0, 1].imshow(median, cmap='gray')
axes[0, 1].set_title('Median Filter')
axes[0, 1].axis('off')

axes[1, 0].imshow(convolve(img, np.ones((5, 5)) / 25), cmap='gray')
axes[1, 0].set_title('Box Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(convolve(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])), cmap='gray')
axes[1, 1].set_title('Horizontal Sobel Filter')
axes[1, 1].axis('off')

axes[2, 0].imshow(convolve(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])), cmap='gray')
axes[2, 0].set_title('Vertical Sobel Filter')
axes[2, 0].axis('off')

gauss = gaussian_kernel(size=5, sigma=1.0)
axes[2, 1].imshow(convolve(img, gauss), cmap='gray')
axes[2, 1].set_title('Gaussian Filter')
axes[2, 1].axis('off')


plt.show()