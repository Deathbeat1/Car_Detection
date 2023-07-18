
import cv2 # Importamos la libreria de OpenCV


cap = cv2.VideoCapture('Deteccion de carros/video4.mp4') # Cargamos el video
object_detector = cv2.createBackgroundSubtractorMOG2() # Cargamos el detector de fondo

while True:
    _, img = cap.read() # Leemos el video
    height, width, _ = img.shape # Obtenemos el alto y ancho del video
    print(height, width)


    #Extraemos la region de interes
    roi = img[300: 720, 150: 900] # Recortamos el video para obtener la region de interes
    

    #1. Objetos detectados
    mask = object_detector.apply(roi) # Aplicamos el detector de fondo
    _, mask = cv2.threshold(mask, 254, 255, 0, cv2.THRESH_BINARY) # Aplicamos un umbral para obtener una imagen binaria
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos de los objetos detectados

    detections = [] # Creamos una lista vacia para guardar las coordenadas de los objetos detectados


    for cnt in contours: # Recorremos los contornos
        
        area = cv2.contourArea(cnt) # Calculamos el area de los contornos
        
        if area > 300: # Si el area es mayor a 300 pixeles 
            x, y, w, h = cv2.boundingRect(cnt) # Obtenemos las coordenadas del contorno
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3) # Dibujamos un rectangulo en el video
            cv2.putText(roi, str('Vehiculo'), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2) # Escribimos el ID del objeto junto con su coordenada
           
            detections.append([x, y, w, h]) # Guardamos las coordenadas de los contornos en la lista

    #2. Mostramos resultados        

    print(detections)
    cv2.imshow('Region', roi) # Mostramos el video
    cv2.imshow('Video con deteccion', img) # Mostramos el video
    cv2.imshow("Contornos", mask) # Mostramos el video con el detector de fondo


    k = cv2.waitKey(60) # Esperamos 60 milisegundos
    if k == 27: # Presionamos ESC para salir
        break

cap.release() # Liberamos la captura
cv2.destroyAllWindows() # Cerramos todas las ventanas
