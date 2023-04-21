# %%
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import time

import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

from micro_dl.input.transformations import apply_affine_transform

# %%
img = cv2.imread(
    "/home/christian.foley/virtual_staining/data_visualization/danes_image_translation/data/01-1m.jpg"
)
img = np.mean(img, -1)  # [100:356, 150:406]
full_img = np.stack((img, img) * 40, 0)

shear_angle = -10
shear_percent = abs(shear_angle / 90)


def apply_window(img, window):
    return img[:, window[0] : window[1], window[2] : window[3]]


window = [120, 360, 200, 540]
window_shape_x = window[-1] - window[-2]

extra_pixels_x = int(window_shape_x * shear_percent)
context_window = [
    window[0],
    window[1],
    window[2] - extra_pixels_x,
    window[3] + extra_pixels_x,
]

final_window = [
    0,
    1000,
    extra_pixels_x * 2,
    2000,
]


small_img = apply_window(full_img, window)
context_img = apply_window(full_img, context_window)
start = time.time()
sheared_context_img = apply_affine_transform(context_img, shear=shear_angle)
print(time.time() - start)
final_img = apply_window(sheared_context_img, final_window)

plt.imshow(small_img[0])
plt.title("small")
plt.show()
plt.imshow(context_img[0])
plt.title("context")
plt.show()
plt.imshow(sheared_context_img[0])
plt.title("sheared context")
plt.show()
plt.imshow(final_img[0])
plt.title("final")
plt.show()

# %%
