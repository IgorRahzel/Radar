# Radar
Este é um projeto que visa implementar a partir do vídeo disponível em `https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK`, o qual captura veículos se deslocando por uma rodavia os seguintes aspectos:

- Identificação e detecção dos veículos
- Calculo das velocidades
- Contagem dos carros para cada uma das vias

## Tabela de Conteúdos
1. Estrutura do Repositório
2. Como Executar
3. Funcionamento do Código

# Extrutura do Repositório:
```bash
Radar
└── src                        # Pasta contendo arquivos .py
    ├── estagio.py             # Arquivo principal           
    └── frame_processor.py     # Arquivo onde é implementada a classe frame_processor
├── imgs                       # Pasta para salvar imagens
├── results                    # Pasta contendo video do resultado obtido
├── README.md                  # README contendo a documentação do projeto  
├── requirements.txt           # Arquivo contendo bibliotecas utilizadas e suas respectivas versões 
```

# Como Executar

1. **Instalar as dependências:**
   Inicialmente certifique-se de que as dependências necessárias foram instaladas, isso pode ser feito executando o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```
2. **Navegue até a pasta do projeto:**
     A partir da pasta do projeto execute o comando:
     ```bash
     python main.py
     ```
3. **Saída:**
   - O vídeo será exibido
   - O vídeo processado será salvo na pasta `results`


# Funcionamento do Código:
Nesta seção serão abordados o funcionamento e aplicação dos métodos da classe `frame_processor`, os métodos serão agrupados de acordo com a sua funcionalidade de modo a facilitar o compreendimento.

O construtor da classe `frame_processor` recebe como argumentos a *ROI* desejada, para esse projeto adotou-se a abordagem de se processar uma *ROI* para a pista da esquerda e outra para a pista da direita da rodovia. As regiões de interesse utilizadas foram armazendas nas variáveis `roii_esquerda` e `roi_direita` como mostrado abaixo:

```python
roi_esquerda = np.array([[0,573],[0,720],[513,716],[641,297],[538,298]], dtype=np.int32)
roi_direita = np.array([[769,720],[677,301],[754,290],[1280,642],[1280,720]], dtype=np.int32)
```

## Processamento dos Frames:
A primeira etapa consiste na execução do método `process_frame` que recebe como parâmetro o frame atual do vídeo. Esse método executa então as seguintes operações:
- Extração da *ROI* do frame, isso é efeito apartir de uma chamada para o método `crop_frame`, o qual basicamente cria uma máscara poligonal e recorta a área delimitada pelo menot retângulo que engloba a *ROI*
- O brilho da imagem é normalizado
- É aplicada a operação de *Background Subtraction*, de modo a tentar capturar apenas a parte não estática do vídeo, i.e os veículos
- São aplicados os filtros Gaussiano e de Mediano visando reduzir o ruído da imagem
- A imagem é convertida para binária através da aplicação de um threshold na intensidade de seus pixels. Essa transformação para binário é necessária para obter um melhor desempenho das operações subsequentes.
- As transformações morfológicas de abertura, fechamento e dilatação são aplicadas na tentativa de remover ainda mais o ruído.
-  Por fim, é feita e extração dos contornos dos veículos.

## Velocidade e Detecção dos veículos:
A velocidade e a detecção dos veículos são implementadas dentro do método `find_speed` o qual tem como argumentos **frame_original**, que é frame que está sendo processado,**dist_threshold** que é a distância máxima a ser considerada para associar um centroide do frame anterior ao atual, **max_speed** esse parâmetro determina a cor do *bounding_box* do veículo, se a velocidade for superior a esse valor a cor é vermelho, caso contrário é verde.

O cálculo da velocidade é feito com base no centróides dos veículos considerando o deslocamento realizado do frame anterior para o atual. O metódo `find_centroids` é responsável por atualizar as listas *centroides_atual*, a qual armazena os centroides do frame atual, e *centroides_anterior* responsável por armazenar os centroides computados no frame anterior.
Os centroídes são calculados da seguinte forma:
Para cada um dos contornos identificados são extraídos os retângulos que limitam esses contornos através da função 
```python 
cv2.boundingRect()
```
O centroíde é então definido como sendo o ponto médio desse retângulo. Voltando então para o método `find_speed` calculamos a distância euclidiana entre os centróides do frame atual e anterior e caso ele seja menor que o threshold estabelecido é então calculada a distância. Como a distânia é calculada entre dois frames subsequentes temos que a velocidade em *pixels/frame* tem o mesmo módulo da distância.

## Contagem dos veículos:
A contagem dos veículos é implementada exclusivamente pelo método `count_vehicles`, nele são passados como parâmetros **frame**, que representa o frame atual, **line** que são os pontos da extremidade de um seguimento de reta, **epsilon** que é a distância entre o segundo seguimento de reta a ser criado em relação àquele informado no parâmetro anterior.
Esse método então consite na criação de dois segmentos de reta de acordo com os parâmetros fornecidos e então quando o centróide de um veículo se encontra entre esses dois segmentos é feita a contagem do veículo.
