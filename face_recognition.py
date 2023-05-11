#sistem kütüphanelerini tanımladım
import cv2
import numpy as np 

identifier = cv2.face.LBPHFaceRecognizer_create() #yüz tanıyıcı oluşturdum
identifier.read('demo/demo.yml') #okunacak dosyayı gösterdim
cascadePath = "haarcascade_frontalface_default.xml" #indirdiğim sınıflandırıcı xml dosyasını handle ettim.

faceCascade = cv2.CascadeClassifier(cascadePath); #kullanılacak yolu atadım
font = cv2.FONT_HERSHEY_SIMPLEX #yazı tipini belirledim
vid_cam = cv2.VideoCapture(0) #bilgisayar kamerası tanındı , harici kamera için VideoCapture(1) yapılmalı

while True:
 
    ret, kamera =vid_cam.read() #kamerayı okuttum
    gray = cv2.cvtColor(kamera,cv2.COLOR_BGR2GRAY) #gri ton ekledim,renkli resmi önce gri tona çevirmem gerekli taramadan önce
    faces = faceCascade.detectMultiScale(gray, 1.2,5)  #sınırlandırmaları tanımladım
 
    for(x,y,w,h) in faces:

        cv2.rectangle(kamera, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4) #frame kalınlığı ve rengi ident edildi

        Id,conf = identifier.predict(gray[y:y+h,x:x+w]) #yazılacak isim değişkeni tanıtıldı
        print(Id)
        
        if(Id == 1):
            Id = "Furkan" #birinci yüz ise Furkan yaz
        
        elif(Id == 2): 
            Id = "Orhan"  #ikinci yüz ise Orhan yaz
            
        elif(Id == 3):
            Id = "Yalova" #üçüncü yüz ise Yalova yaz
        

        cv2.rectangle(kamera, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1) #frame boyutları ayarlandı
        cv2.putText(kamera, str(Id), (x,y-40), font, 2, (255,255,255), 3) #yazılacak isim boyutu,rengi ve kalınlığı belirlendi
    
    cv2.imshow('kamera',kamera) #kamera göster komutu ekledim

    if cv2.waitKey(10) & 0xFF == ord('q'): #çıkış tuşu atadım
        break
    

vid_cam.release() #kamera stop edilir

cv2.destroyAllWindows() #tüm pencereler kapatılır