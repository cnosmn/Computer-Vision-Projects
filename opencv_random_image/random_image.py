import numpy as np
import cv2

# Rastgele sayıları içeren bir matris oluştur
random_matrix = np.random.randint(0, 180, size=(800, 800), dtype=np.uint8)

# # Matrisi OpenCV ile göster
# cv2.imshow('Random Matrix', random_matrix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Matrisi 0-155 arasındakileri siyah, diğerlerini beyaz yap
threshold_value = 155
ret, thresholded_matrix = cv2.threshold(random_matrix, threshold_value, 255, cv2.THRESH_BINARY)

# Sonucu göster
cv2.imshow('Thresholded Matrix', thresholded_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()


