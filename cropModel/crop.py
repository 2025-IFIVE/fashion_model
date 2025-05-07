import cv2
import numpy as np
import rembg


MAX_IMAGE_SIZE = 512  


def resize_image_if_large(image, max_size=MAX_IMAGE_SIZE):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_w, new_h = int(width * scale), int(height * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def apply_grabcut(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = resize_image_if_large(image)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = image.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    result[:, :, 3] = mask2 * 255

    return result
