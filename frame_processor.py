import numpy as np
import cv2

class frame_processor:
    def __init__(self,ROI,fps):
        self.vehicle_counter = 0
        self.ROI = ROI
        self.backSub = cv2.createBackgroundSubtractorKNN(history=200,dist2Threshold=100.0,detectShadows=False)
        self.centroides_atual = []
        self.tempo_por_frame = 1/fps   

    def crop_frame(self,frame):
        # Extrair a região de interesse
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [self.ROI], (255, 255, 255))
        cropped = cv2.bitwise_and(frame, mask)
        x, y, w, h = cv2.boundingRect(self.ROI)
        final_crop = cropped[y:y+h, x:x+w]
        return final_crop
    
    def morphological_operations(self,frame):
        # Aplicar operações morfológicas para remover ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # kernel 3x3
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel) # operação de abertura
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel) # operação de fechamento
        frame = cv2.dilate(frame, kernel, iterations=1) # dilatação
        return frame

    def process_frame(self,frame):
        # retirar ROI
        frame = self.crop_frame(frame)
        # Normalizar brilho
        normalized_frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # Background Subtraction
        frame_BS = self.backSub.apply(normalized_frame)
        # Aplicar filtro para remover ruído (Filtro Gaussiano)
        frame_BS = cv2.GaussianBlur(frame_BS, (5, 5), 0)
        # Aplicar filtro para remover ruído (Filtro de Mediana)
        frame_BS = cv2.medianBlur(frame_BS,5)
        # Apply threshold to convert to binary
        _,frame_binario = cv2.threshold(frame_BS, 220, 255, cv2.THRESH_BINARY)
        # Morphological operations to remove noise
        self.frame_binario = self.morphological_operations(frame_binario)
        # Find contours of vehicles
        self.contours,_ = cv2.findContours(self.frame_binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove Intersections
    def isInside(self, coordinates, x, y, w, h):
        for coordinate in coordinates:
            x_1, y_1, w_1, h_1 = coordinate
            # Confere se alguma bounding box antiga está dentro da atual
            if x < x_1 and (x + w) < (x_1 + w_1) and y < y_1 and (y + h) < (y_1 + h_1):
                return True
            # Confere se a bounding box do contorno atual está dentro de algum prévio
            if x > x_1 and (x + w) > (x_1 + w_1) and y > y_1 and (y + h) > (y_1 + h_1):
                return True
        return False
    
    def find_centroids(self,min_area = 3000):
        self.centroide_anterior = self.centroides_atual.copy()
        self.centroides_atual = []
        
        # Filtrar contornos com base na área mínima
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) >= min_area]
    

        # Lista para armazenar as coordenadas finais
        coordinates = []
    
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Verificar se o contorno atual está dentro de outro contorno
            #if not self.isInside(coordinates, x, y, w, h):
            # Calcular o centróide
            cx = x + w // 2
            cy = y + h // 2
            # Adicionar as coordenadas do bounding box
            coordinates.append([x, y, w, h])
            # Adicionar o centróide à lista
            self.centroides_atual.append((cx, cy))
        return self.centroides_atual

    
    def find_speed(self,frame_original,dist_threshold = 50,max_speed = 100):
        # Encontrar a velocidade e imprimi-la na imagem em cima do bounding box
        for i, atual in enumerate(self.centroides_atual):
            cx_atual, cy_atual = atual
            for anterior in self.centroide_anterior:
                cx_anterior, cy_anterior = anterior
                # Calcular a distância entre os centroides atual e anterior
                distancia = np.sqrt((cx_atual - cx_anterior)**2 + (cy_atual - cy_anterior)**2)
                # Se a distância for menor que o threshold, calcular a velocidade
                if distancia < dist_threshold:
                    velocidade = distancia/self.tempo_por_frame
                    # Criar bounding box ao redor do veiculo com a velocidade
                    x_offset, y_offset, _,_ = cv2.boundingRect(self.ROI)
                    x, y, w, h = cv2.boundingRect(self.contours[i])
                    x_original = x + x_offset
                    y_original =  y + y_offset
                    # Desenhar Retângulo no frame_original
                    # Alterar a cor do bounding box com base na velocidade
                    cor_bounding_box = (0, 255, 0) if velocidade <= max_speed else (0, 0, 255)

                    # Desenhar o bounding box no frame original
                    cv2.rectangle(frame_original, (x_original, y_original), 
                              (x_original + w, y_original + h), cor_bounding_box, 2)

                    # Exibir a velocidade acima do bounding box
                    cv2.putText(frame_original, f'{velocidade:.2f} px/s', 
                            (x_original, y_original - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_bounding_box, 2)
                    break

    def count_vehicles(self,frame,line,epsilon,counter_placement,pista,counter_height = 400):
        x1, y1, x2, y2 = line

        # Coordenadas da linha deslocada
        x3, y3, x4, y4 = x1, y1 + epsilon, x2, y2 + epsilon

        #cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.05)  # Linha verde com seta indicando a direção
        # Linha deslocada por epsilon
        #x3, y3 = x1, y1 + epsilon
        #x4, y4 = x2, y2 + epsilon
        #cv2.arrowedLine(frame, (x3, y3), (x4, y4), (255, 0, 0), 2, tipLength=0.05)  # Linha azul com seta indicando a direção
        
        x_offset, y_offset, _,_ = cv2.boundingRect(self.ROI)
        # Iterar sobre os centróides detectados
        for centroid in self.centroides_atual:
            cx, cy = centroid
            cx,cy = cx + x_offset, cy + y_offset
            # Verificar se o centróide está à esquerda da primeira linha
            det1 = (cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)
            if det1 < 0:
                lado_linha1 = 0
            else:
                lado_linha1 = 1


            # Verificar se o centróide está à direita da segunda linha deslocada
            det2 = (cx - x3) * (y4 - y3) - (cy - y3) * (x4 - x3)

            if det2 < 0:
                lado_linha2 = 0
            else:
                lado_linha2 = 1

            
            if lado_linha1 != lado_linha2:
                self.vehicle_counter += 1


        cv2.putText(frame,f'Veiculos {pista}: {(self.vehicle_counter)}',counter_placement,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)