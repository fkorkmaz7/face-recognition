#kütüphaneleri tanımladım
import cv2

vid_cam = cv2.VideoCapture(0) #kamerayı tanımladım

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#sınıflandırıcıyı dahil ettim 
face_id = 3 # tanımlanan her yüz için farklı id atadım

count = 0 #çekilecek resim adedi için counter tanımladım

while(True):

    _, image_frame = vid_cam.read() #kamera okuttum

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY) #resim rengi için gri tonlama ekledim

    faces = detector.detectMultiScale(gray, 1.3, 5) #resim için alt ve üst sınırlar belirledim 
    
    for (x,y,w,h) in faces:

        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2) #frame kalınlığı ve rengi belirledim
        
        count += 1 #fotoğraf adet artışını tanımladım

        cv2.imwrite("database/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        #resimleri veri klasörüne istenilen formatta yazdırdım.
        cv2.imshow('frame', image_frame) #kamerayı göster komutu atadım
        
    if cv2.waitKey(100) & 0xFF == ord('q'): #kamerayı kapatma tuşu atadım
        break

    elif count>100: #maksimum 100 resim aktarması ile kısıtladım.
        break
    

vid_cam.release() #kamerayı kapat 

cv2.destroyAllWindows() #tüm pencereleri durdur