from __future__ import print_function
import cv2
from cap_from_youtube import cap_from_youtube
import numpy as np
import os
from utils import crop_frame



# Resolver problemas de plugins do OpenCV
os.environ["QT_QPA_PLATFORM"] = "xcb"
cv2.setNumThreads(1)

# Background Subtraction
backSub_direita = cv2.createBackgroundSubtractorKNN(history=200)
backSub_esquerda = cv2.createBackgroundSubtractorKNN(history=200)

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
    # Aplicar o Background Subtraction
    crop_esquerda_BS = backSub_esquerda.apply(crop_esquerda)
    
    # Remover ruido
    #denoised_esquerda = cv2.fastNlMeansDenoising(crop_esquerda_BS, None, 10, 7, 21)
    
    _,crop_esquerda_binario = cv2.threshold(crop_esquerda_BS, 220, 255, cv2.THRESH_BINARY)
    
    #Encontrar contorno dos veículos
    contours_esq,_ = cv2.findContours(crop_esquerda_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_esquerda_binario, contours_esq, -1, (0,255,0), 3)
    




    # Definir os pontos do ROI para a pista da direita
    roi_direita = np.array([[676,290], [822,290], [1180, 540], [711, 540]], dtype=np.int32)
    crop_direita = crop_frame(frame, roi_direita)
    crop_direita_BS = backSub_direita.apply(crop_direita)
    # Remover ruido
    #denoised_direita = cv2.fastNlMeansDenoising(crop_direita_BS, None, 10, 7, 21)
    __,crop_direita_binario = cv2.threshold(crop_direita_BS, 220, 255, cv2.THRESH_BINARY)
    #Encontrar contorno dos veículos
    contours_dir,_ = cv2.findContours(crop_direita_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_direita_binario, contours_dir, -1, (0,255,0), 3)
    # Reduzir o tamanho da janela do vídeo original
    small_frame = cv2.resize(frame, (640, 360))  # Tamanho ajustável

    # Mostrar os resultados
    cv2.imshow("Frame Original (Reduzido)", small_frame)
    cv2.imshow("Recorte Diagonal - Esquerda", crop_esquerda_binario)
    cv2.imshow("Recorte Diagonal - Direita", crop_direita_binario)

    keyboard = cv2.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:  # Tecla 'q' ou 'ESC' para sair
        break

cv2.destroyAllWindows()


### usar morphological operation

'''
1 - separar backgroud -> imagem escala cinza
2 - treshould -> imagem binária
3 - find contours -> segmentação
4 - calcular os centróids de cada carro
5 - ter numa lista de vetores os centróides de cada carro (frame atual e pro frame anterior)
'''