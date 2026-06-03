"""
Custom object detector for Kalman Filter tracking.

Este módulo contém o código de detecção de objetos que será utilizado
pelo sistema de Kalman para gerar as medições de posição.

INSTRUÇÕES DE USO:
1. Implemente sua própria função detect_centroid(frame)
2. Ela deve retornar (cX, cY) ou None se nenhum objeto detectado
3. A função será chamada automaticamente no processamento e playback

EXEMPLO:
    def detect_centroid(frame):
        # Seu código aqui
        # ...
        return (x, y)  ou None
"""

import cv2
import numpy as np


def detect_centroid(frame):
    """
    Detecta o centrado de um objeto no frame.
    
    Args:
        frame: numpy array BGR (formato OpenCV)
        
    Returns:
        Tupla (cX, cY) com coordenadas do centróide
        None se nenhum objeto detectado
        
    CUSTOMIZE ESTE MÉTODO COM SEU DETECTOR!
    """
    # Exemplo padrão: limiarização + maior contorno
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 25:
        return None
    
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def detect_color_red(frame):
    '''
        Código para detecção de um objeto de uma determinada cor na figura.
    '''
    cX = 1 
    cY = 1

    return (cX, cY)