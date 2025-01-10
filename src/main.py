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

tempo_total = 0

# Configurações para salvar o vídeo
output_filename = 'results/resultado.mp4'
frame_width = 640  # Resolucão compatível
frame_height = 360
duração = 5
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Ou 'avc1'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
Recording = False


if not capture.isOpened():
    print('Unable to open')
    exit(0)

# Inicializando os processadores de frame
frame_processor_direita = frame_processor(roi_direita)
frame_processor_esquerda = frame_processor(roi_esquerda)


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


    # Encontrar a velocidade
    frame_processor_direita.find_speed(frame)
    frame_processor_esquerda.find_speed(frame)

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

    if keyboard == ord('r'):
        Recording = True
    
    if Recording:
        # Salvar o frame processado no vídeo
        out.write(small_frame)
        tempo_total += tempo_por_frame
        print(tempo_total)
        if tempo_total >= 60:
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