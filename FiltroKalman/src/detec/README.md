# Detector Customizável - src/detec/

Este diretório contém o código de detecção de objetos que será utilizado pelo Kalman Filter.

## 📁 Estrutura

```
src/detec/
├── __init__.py
└── detector.py      # Seu código de detecção aqui
```

## 🎯 Como Customizar

### Edite `detector.py`

A função `detect_centroid(frame)` será chamada automaticamente durante:
1. **Processamento**: Cada frame do vídeo é processado
2. **Playback**: Quando toca o vídeo (sincronização com gráficos)

### Assinatura da Função

```python
def detect_centroid(frame):
    """
    Detecta o centroide de um objeto no frame.
    
    Args:
        frame: numpy array BGR (formato OpenCV) - shape (H, W, 3)
        
    Returns:
        (cX, cY): tupla com coordenadas do centroide em pixels
        None: se nenhum objeto detectado
    """
    # Seu código aqui
    ...
    return (x, y)  # ou None
```

## 💡 Exemplos

### Exemplo 1: Detecção por Cor (HSV)

```python
import cv2
import numpy as np

def detect_centroid(frame):
    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range de cor (ajuste conforme sua cor)
    lower_color = np.array([10, 100, 100])
    upper_color = np.array([20, 255, 255])
    
    # Máscara
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Encontrar contorno
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Centóide do maior contorno
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    
    if M["m00"] == 0:
        return None
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)
```

### Exemplo 2: Detecção com Blob

```python
import cv2

def detect_centroid(frame):
    # Setup SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detecção
    keypoints = detector.detect(frame)
    
    if not keypoints:
        return None
    
    # Maior blob
    largest_kp = max(keypoints, key=lambda kp: kp.size)
    return (int(largest_kp.pt[0]), int(largest_kp.pt[1]))
```

### Exemplo 3: Template Matching

```python
import cv2
import numpy as np

# Variável global (carregue uma vez)
template = None

def detect_centroid(frame):
    global template
    
    if template is None:
        # Carregar template (ajuste o caminho)
        template = cv2.imread("seu_template.png", cv2.IMREAD_GRAYSCALE)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    # Calcular centróide
    h, w = template.shape
    cX = max_loc[0] + w // 2
    cY = max_loc[1] + h // 2
    
    return (cX, cY)
```

## ⚙️ Melhorias Recomendadas

- **Estabilidade**: Filtro de média móvel para reduzir jitter
- **Rastreamento**: Usar centroid anterior para ajudar na busca
- **ROI**: Procurar apenas em região de interesse
- **Confiança**: Retornar confidence junto com (x, y)

## 🔗 Integração Automática

O sistema chamará `detect_centroid(frame)` automaticamente:

```python
from src.detec.detector import detect_centroid

# No processamento:
for frame in video:
    measurement = detect_centroid(frame)  # Sua função aqui!
    kalman.update(measurement)
```

Você **não precisa modificar** `app.py` - a integração é automática! 🚀

## 📝 Debug

Se seu detector não está funcionando:

1. Teste manualmente em um script:
```python
import cv2
from src.detec.detector import detect_centroid

cap = cv2.VideoCapture("seu_video.mp4")
ret, frame = cap.read()
result = detect_centroid(frame)
print(f"Resultado: {result}")
```

2. Verifique o console durante processamento
3. Ative checkboxes de debug para visualizar detecção

---

**Dúvidas?** Consulte a documentação do OpenCV: https://docs.opencv.org/
