import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)
gauss = cv2.GaussianBlur(img, (5, 5), 0)

kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(img, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(img, -1, kernel_roberts_y)
roberts = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

kernel_prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernel_prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_x = cv2.filter2D(img, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(img, -1, kernel_prewitt_y)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

canny = cv2.Canny(img, 100, 200)


filtros = [
    ('Original', img), ('Gaussiano', gauss), 
    ('Roberts', roberts), ('Prewitt', prewitt),
    ('Sobel', sobel), ('Laplacian', laplacian), ('Canny', canny)
]

plt.figure(figsize=(16, 10))

for i, (nome, imagem) in enumerate(filtros):
    plt.subplot(2, 4, i+1)
    plt.imshow(imagem, cmap='gray')
    plt.title(nome)
    plt.axis('off')

plt.tight_layout()
plt.show()