from __future__ import print_function
import cv2
from cap_from_youtube import cap_from_youtube
import numpy as np
import os
from utils import crop_frame



# Resolver problemas de plugins do OpenCV
os.environ["QT_QPA_PLATFORM"] = "xcb"
cv2.setNumThreads(1)

# Captura do vídeo
capture = cap_from_youtube('https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK', '720p')

if not capture.isOpened():
    print('Unable to open')
    exit(0)

while True:
    # Leitura do frame
    ret, frame = capture.read()
    if frame is None:
        break

    # Definir os pontos do ROI para a pista da esquerda
    roi_esquerda = np.array([[541, 294], [648, 294], [574, 523], [76, 523]], dtype=np.int32)
    crop_esquerda = crop_frame(frame, roi_esquerda)

    # Definir os pontos do ROI para a pista da direita
    roi_direita = np.array([[676,290], [822,290], [1180, 540], [711, 540]], dtype=np.int32)
    crop_direita = crop_frame(frame, roi_direita)

   
    # Reduzir o tamanho da janela do vídeo original
    small_frame = cv2.resize(frame, (640, 360))  # Tamanho ajustável

    # Mostrar os resultados
    cv2.imshow("Frame Original (Reduzido)", small_frame)
    cv2.imshow("Recorte Diagonal - Esquerda", crop_esquerda)
    cv2.imshow("Recorte Diagonal - Direita", crop_direita)

    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:  # Tecla 'q' ou 'ESC' para sair
        break

cv2.destroyAllWindows()
