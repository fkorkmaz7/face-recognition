#gerekli kütüphaneleri ekledim
import cv2
import os
import numpy as np
from PIL import Image

identifier = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#yüz tanımak ve sınıflandırma yapmak için yüz tanıma ve xml dosyası ekledim

def getImagesAndLabels(path): # resim ve tanımlar için yol atadım (path = yol)
    
    ImagePaths = [os.path.join(path,f) for f in os.listdir(path)] # resim yolu için döngü tanımladım
    
    face_sample = [] # degisken1 = benzer yüzler 
    
    names = []  # değişken2 = yüz isimler
    
    for ImagePath in ImagePaths:
        
        PIL_img = Image.open(ImagePath).convert('L') #resmin okunacağı dosyayı açmak için PIL kütüphanesini kullandım
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(ImagePath)[-1].split(".")[1]) #isim nereye yazılsın?
        print(id)
        faces = detector.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in faces:
         face_sample.append(img_numpy[y:y+h,x:x+w]) # karakter dizisini belirledim 
         names.append(id)  # karakter dizisini okuttum

    return face_sample,names # yüz ve yüz isimleri döndürdüm

faces, names = getImagesAndLabels('database') #resimlerin bulunduğu klasör adını girdim
identifier.train(faces, np.array(names)) #tanıyıcı demo oluşturdum
identifier.save('demo/demo.yml') #save klasörü belirledim, çekilen fotolar için. 
        
