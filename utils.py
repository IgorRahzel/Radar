import numpy as np
import cv2
def crop_frame(frame,roi):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi], (255, 255, 255))
    cropped = cv2.bitwise_and(frame, mask)
    x, y, w, h = cv2.boundingRect(roi)
    final_crop = cropped[y:y+h, x:x+w]
    return final_crop

def associate_centroids(centroide_anterior,centroides_atual):
    # Associar os centroides atuais com os centroides anteriores
    for i in range(len(centroide_anterior)):
        for j in range(len(centroides_atual)):
            if np.linalg.norm(np.array(centroide_anterior[i])-np.array(centroides_atual[j]))<10:
                centroide_anterior[i] = centroides_atual[j]
                break
    return centroide_anterior
