import numpy as np
import cv2
def crop_frame(frame,roi):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi], (255, 255, 255))
    cropped = cv2.bitwise_and(frame, mask)
    x, y, w, h = cv2.boundingRect(roi)
    final_crop = cropped[y:y+h, x:x+w]
    return final_crop