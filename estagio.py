from __future__ import print_function
import cv2
from cap_from_youtube import cap_from_youtube
import numpy as np
import os
from frame_processor import frame_processor
roi_direita = np.array([[676,290], [822,290], [1180, 540], [711, 540]], dtype=np.int32)
roi_esquerda = np.array([[541, 294], [648, 294], [574, 523], [76, 523]], dtype=np.int32)

# Resolver problemas de plugins do OpenCV
os.environ["QT_QPA_PLATFORM"] = "xcb"
cv2.setNumThreads(1)

# Captura do vídeo
capture = cap_from_youtube('https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK', '720p')

# fps do vídeo
fps = capture.get(cv2.CAP_PROP_FPS)
tempo_por_frame = 1/fps

if not capture.isOpened():
    print('Unable to open')
    exit(0)

frame_processor_direita = frame_processor(roi_direita,fps)
frame_processor_esquerda = frame_processor(roi_esquerda,fps)


while True:
    # Leitura do frame
    ret, frame = capture.read()
    if frame is None:
        break


    # Processamento do frame
    frame_processor_direita.process_frame(frame)
    frame_processor_esquerda.process_frame(frame)

    # Encontrar os centroides
    centroides_direita = frame_processor_direita.find_centroids()
    centroides_esquerda = frame_processor_esquerda.find_centroids()
    
    small_frame = cv2.resize(frame, (640, 360))  # Tamanho ajustável

    # Mostrar os resultados
    cv2.imshow("Frame Original (Reduzido)", small_frame)
    cv2.imshow("Recorte Diagonal - Esquerda", frame_processor_esquerda.frame_binario)
    cv2.imshow("Recorte Diagonal - Direita", frame_processor_direita.frame_binario)

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