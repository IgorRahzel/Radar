import numpy as np
import cv2

class frame_processor:
    def __init__(self,ROI,fps):
        self.ROI = ROI
        self.backSub = cv2.createBackgroundSubtractorKNN(history=200,detectShadows=False)
        self.centroides_atual = []
        self.tempo_por_frame = 1/fps   

    def crop_frame(self,frame):
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [self.ROI], (255, 255, 255))
        cropped = cv2.bitwise_and(frame, mask)
        x, y, w, h = cv2.boundingRect(self.ROI)
        final_crop = cropped[y:y+h, x:x+w]
        return final_crop
    
    def morphological_operations(self,frame):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        frame = cv2.dilate(frame, kernel, iterations=1)
        return frame

    def process_frame(self,frame):
        # Background Subtraction
        frame_BS = self.backSub.apply(self.crop_frame(frame))
        # Apply threshold to convert to binary
        _,frame_binario = cv2.threshold(frame_BS, 220, 255, cv2.THRESH_BINARY)
        # Morphological operations to remove noise
        self.frame_binario = self.morphological_operations(frame_binario)
        # Find contours of vehicles
        self.contours,_ = cv2.findContours(self.frame_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    def find_centroids(self):
        self.centroide_anterior = self.centroides_atual.copy()
        self.centroides_atual = []
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w//2
            cy = y + h//2
            # Desenhar bounding box
            cv2.rectangle(self.frame_binario, (x, y), (x + w, y + h), (255, 255,255), 2)
            self.centroides_atual.append((cx, cy))
        return self.centroides_atual
    
  