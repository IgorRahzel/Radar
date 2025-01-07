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
backSub_direita = cv2.createBackgroundSubtractorKNN(history=200,detectShadows=False)
backSub_esquerda = cv2.createBackgroundSubtractorKNN(history=200,detectShadows=False)

# Criando lista de centroides atuais
centroides_atual = []

# Captura do vídeo
capture = cap_from_youtube('https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK', '720p')

# fps do vídeo
fps = capture.get(cv2.CAP_PROP_FPS)
tempo_por_frame = 1/fps

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
    
    # Aplicar threshold na imagem para converter para binário
    _,crop_esquerda_binario = cv2.threshold(crop_esquerda_BS, 220, 255, cv2.THRESH_BINARY)
    
    # Realizar operações morfológicas para remover ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel ajustável (5x5)
    crop_esquerda_binario = cv2.morphologyEx(crop_esquerda_binario, cv2.MORPH_OPEN, kernel) # Abertura: Remove pequenos ruídos brancos 
    crop_esquerda_binario = cv2.morphologyEx(crop_esquerda_binario, cv2.MORPH_CLOSE, kernel) # Fechamento: Preenche pequenas lacunas dentro dos objetos detectados
    crop_esquerda_binario = cv2.dilate(crop_esquerda_binario, kernel, iterations=1) # dilatação para aumentar os contornos e facilitar a detecção

    #Encontrar contorno dos veículos
    contours_esq,_ = cv2.findContours(crop_esquerda_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop_esquerda_binario, contours_esq, -1, (0,255,0), 3)

    #listas para armazenar os centroides
    centroide_anterior = centroides_atual.copy()
    centroides_atual = []
    

    # Encontrar centróides e bounding box dos veículos
    for contour in contours_esq:
        # Encontrar bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Calcular centroides
        cx = x + w//2
        cy = y + h//2
        centroides_atual.append((cx, cy))
        # Desenhar bounding box
        cv2.rectangle(crop_esquerda_binario, (x, y), (x+w, y+h), (255, 0, 255), 2)



    # Definir os pontos do ROI para a pista da direita
    roi_direita = np.array([[676,290], [822,290], [1180, 540], [711, 540]], dtype=np.int32)
    crop_direita = crop_frame(frame, roi_direita)
    crop_direita_BS = backSub_direita.apply(crop_direita)
    # Aplicar threshold na imagem para converter para binário
    __,crop_direita_binario = cv2.threshold(crop_direita_BS, 220, 255, cv2.THRESH_BINARY)
    # Realizar operações morfológicas para remover ruídos

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
        print(contours_esq)
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