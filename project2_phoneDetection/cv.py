from ultralytics import YOLO
# Burada kullanacağımız modeli seçiyoruz.
model= YOLO("yolov8l.pt") 

# Gerekli kütüphaneleri dahil ediyoruz. 
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


kamera= cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# El tespitinde kullanacağımız modeli tanımlıyoruz
detector=HandDetector(maxHands=5)

while True:
    ret,kare=kamera.read()
    if not ret:
        break
    # videoda bulunan insanların konumunu tutan liste
    person_list=[]
    
    # videoda bulunan telefonların  konumunu tutan liste
    phone_list=[]
    
    # Resmi RGB formata çevirip nesne nesne tespit modeline veriyoruz.
    imgs=cv2.cvtColor(kare,cv2.COLOR_BGR2RGB)
    results = model(imgs,verbose=False) 
    labels=results[0].names
    
    
    for i in range(len(results[0].boxes)):
        x1,y1,x2,y2=results[0].boxes.xyxy[i]
        score=results[0].boxes.conf[i]
        label=results[0].boxes.cls[i]
        x1,y1,x2,y2,score,label=int(x1),int(y1),int(x2),int(y2),float(score),int(label)
        name=labels[label]
        
        # %50'nin altında bulunan nesneleri göz ardı ediyoruz.
        if score<0.5:
            continue
        # Eğer nesne insan ise bu nesnenin konumunu gerekli listenin içine ekliyoruz.
        if name=='person':
            
            person_list.append((x1,y1,x2,y2))
            
        # Aynısını telefon için yapıyoruz.
        if name=='cell phone':
            
            phone_list.append((x1,y1,x2,y2))  
          
            

    # Burada videodaki karenin kopyası oluşturuluyor.
    # Bunun sebebi el tespitinde yapılan işlemler orijinal görseli etkilemesin diye
    copy=kare.copy() 
    
    # Burada kopya görseli el tespit modeline veriyoruz.
    hands,copy=detector.findHands(copy,flipType=False)
    
    # Resimdeki el ile kesişen telefonların orta noktaların konumunu bununla tutuyoruz
    hand_list=[]
    
    # Burada her bir telefon için resimde bir bölge oluşturacağız.
    for phone in phone_list:
        (x21,y21,x22,y22)=phone
        region1=np.array([(x21,y21),(x22,y21),(x22,y22),(x21,y22)])
            
        region1 = region1.reshape((-1,1,2))
        
        # Burada her bir el için eldeki tüm noktalara bakacağız.
        for hand in hands:
            # 21 deme sebebimiz elde 21 adet nokta bulunması
            for j in range(21):
                # Her bir konumu sırayla alıyoruz.
                x,y,z=hand['lmList'][j]
                # Her bir nokta için bu noktanın telefonun olduğu bölgenin içinde olup olmadığına bakıyoruz.
                inside_region1=cv2.pointPolygonTest(region1,(x,y),False)
                # Eğer elin bir noktası telefonun olduğu bölgenin içinde ise o telefonun orta noktasını uygun listeye ekliyoruz 
                if inside_region1>0:
                    cx=int(x21/2+x22/2)
                    cy=int(y21/2+y22/2)
                    hand_list.append((cx,cy))
                    
    # Burada ise her görseldeki her bir kişi için bir bölge oluşturup 
    # el ile kesişen telefonun orta noktası var mı diye bakıyoruz. 
    # Eğer var ise control değişkenini true yapıyoruz.
    for person in person_list:
        control=False
        (x21,y21,x22,y22)=person
        region1=np.array([(x21,y21),(x22,y21),(x22,y22),(x21,y22)])
            
        region1 = region1.reshape((-1,1,2))
        
        for hand in hand_list:
                (x,y)=hand
                inside_region1=cv2.pointPolygonTest(region1,(x,y),False)
                if inside_region1>0:
                    control=True
                   
        # Eğer bu kişi telefon ile uğraşıyorsa kare içine alınıp aşağıdaki yazı yazılıyor.
        if control:
            cv2.rectangle(kare,(x21,y21),(x22,y22),(102,0,153),5)
            cv2.putText(kare, 'Phone Detected',(x21, y21-20), font, 2, (255,0,0), 2)
    # kare = cv2.resize(kare, (0, 0), fx = 0.4, fy = 0.4)
    #kare = cv2.resize(kare,(500,500))
    print(kare.shape)
    cv2.imshow("kamera",kare)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()