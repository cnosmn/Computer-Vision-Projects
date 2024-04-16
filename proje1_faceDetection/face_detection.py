import cv2
import serial

# Arduino seri portu bağlantısı
#ser = serial.Serial('COM4', 9600)  # COMX, Arduino'nun bağlı olduğu seri portu temsil eder

# Yüz tanıma modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera bağlantısı
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Eğer yüz tespit edildiyse
    # if len(faces) > 0:
    #     # Arduino'ya LED'i yakmasını söyle
    #     ser.write(b'1')
    # else:
    #     # Arduino'ya LED'i söndürmesini söyle
    #     ser.write(b'0')

    # Görüntüyü göster
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Seri bağlantıyı kapat
ser.close()
# Kamera bağlantısını kapat
cap.release()
cv2.destroyAllWindows()
