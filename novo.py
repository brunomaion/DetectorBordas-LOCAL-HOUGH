import numpy as np
import cv2

def process_local_edges(input_image, threshold_magnitude, target_orientation, tolerance_orientation):
    # Passo 1: Calcular a magnitude e a orientação do gradiente
    magnitude, orientation = calculate_gradient(input_image)
    
    # Passo 2: Criar a imagem binária g(x, y)
    binary_image = np.zeros_like(magnitude, dtype=np.uint8)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] > threshold_magnitude and is_in_orientation_range(orientation[i, j], target_orientation, tolerance_orientation):
                binary_image[i, j] = 255
                
    return binary_image

def calculate_gradient(input_image):
    dx = cv2.Sobel(input_image, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(input_image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(dx*2 + dy*2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convertendo para graus
    return magnitude, orientation

def is_in_orientation_range(angle, target_angle, tolerance):
    return np.abs(angle - target_angle) <= tolerance

# Exemplo de uso:
input_image = cv2.imread('CamMan.bmp', cv2.IMREAD_GRAYSCALE)
threshold_magnitude = 100  # Limiar de magnitude do gradiente
target_orientation = 0  # Orientação alvo
tolerance_orientation = 10  # Tolerância de orientação em graus

binary_image = process_local_edges(input_image, threshold_magnitude, target_orientation, tolerance_orientation)
cv2.imshow('Imagem binária', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()