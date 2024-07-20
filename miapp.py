import cv2
import numpy
#agregar el import os para leer los directorios
import os


#faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



#faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#modificar la linea para anadir el directorio del xml 
haarcascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
#ahora el faceClassif tiene se le pasa la ruta del xml 
faceClassif = cv2.CascadeClassifier(haarcascade_path)



#image = cv2.imread('C:/A_UTEQ_CICLOS_LECTIVOS/2024 2025 PPA/GCS/PROYECTO_PYTHON/PythonApplication1/fotos_pruebas/caras_3.png')
image = cv2.imread('caras_3.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(30,30),
  maxSize=(200,200))

for (x,y,w,h) in faces:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
