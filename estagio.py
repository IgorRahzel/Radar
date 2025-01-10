from __future__ import print_function
import cv2
from cap_from_youtube import cap_from_youtube
import numpy as np
import os
from frame_processor import frame_processor

# Definindo ROI
roi_esquerda = np.array([[0,573],[0,720],[513,716],[641,297],[538,298]], dtype=np.int32)
roi_direita = np.array([[769,720],[677,301],[754,290],[1280,642],[1280,720]], dtype=np.int32)

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

# Inicializando os processadores de frame
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

    # Coordenadas Homografia
    pts_src = np.array([
    [498, 352],  # Canto superior esquerdo
    [601, 366],  # Canto superior direito
    [478, 710],  # Canto inferior direito
    [68, 687],  # Canto inferior esquerdo
    ], dtype=np.float32)

    # Pontos de Destino
    largura  = 7 # em metros
    comprimento= int(np.max(pts_src[:, 1]) - np.min(pts_src[:, 1]))
    pts_out = np.array([
    [0,0],
    [largura,0],
    [largura,comprimento],
    [0,comprimento]
    ])

    H,_ = cv2.findHomography(pts_src,pts_out)




    # Encontrar a velocidade
    reference_points_direita = (np.array([690,450]),np.array([895,450]))
    reference_points_esquerda = (np.array([260,509]),np.array([549,512]))
    frame_processor_direita.find_speed(frame,reference_points_direita)
    frame_processor_esquerda.find_speed(frame,reference_points_esquerda)

    # Contador de veículos
    counter_placement_esquerda = (120,60)
    counter_placement_direita = (900,60)
    linha_esquerda = (60,560,580,560)
    linha_direita = (700,560,1172,560)
    frame_processor_direita.count_vehicles(frame,linha_direita,15,counter_placement_direita,'direita')
    frame_processor_esquerda.count_vehicles(frame,linha_esquerda,15,counter_placement_esquerda,'esquerda')
    
    small_frame = cv2.resize(frame, (640, 360))  # Tamanho ajustável

    # Mostrar os resultados
    cv2.imshow("Frame Original (Reduzido)", small_frame)
    #cv2.imshow("Recorte Diagonal - Esquerda", frame_processor_esquerda.frame_binario)
    #cv2.imshow("Recorte Diagonal - Direita", frame_processor_direita.frame_binario)

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