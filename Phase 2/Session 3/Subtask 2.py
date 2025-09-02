import numpy as np
import matplotlib.pyplot as plt
import cv2
# Take notice that OpenCV handles the image as a numpy array when opening it 
img = cv2.imread('shapes.jpg')
out = img.copy()

# Make a mask for each color (red, blue, black)

lower_red = np.array([0, 0, 200])
upper_red = np.array([50, 50, 255])
red = cv2.inRange(out, lower_red, upper_red)

lower_black = np.array([0, 0, 0])
upper_black = np.array([80, 80, 80])
black = cv2.inRange(out, lower_black, upper_black)

lower_blue = np.array([200, 0, 0])
upper_blue = np.array([255, 50, 50])
blue = cv2.inRange(out, lower_blue, upper_blue)

# Take care that the default colorspace that OpenCV opens an image in is BGR not RGB

# Change all pixels that fit within the blue mask to black
out[blue > 0] = [0, 0, 0]
# Change all pixels that fit within the red mask to blue
out[red > 0] = [255, 0, 0]
# Change all pixels that fit within the black mask to red
out[black > 0] = [0, 0, 255]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(out_rgb)
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()