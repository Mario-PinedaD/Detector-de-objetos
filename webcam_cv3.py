import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "carterav5_mimagenes.xml"
objetoCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename = "webcam.log", level = log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
cant = 0

while True:
    if not video_capture.isOpened():
        print("Unable to load camera.")
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    objetos = objetoCascade.detectMultiScale(
        gray,
        # scaleFactor = 1.1,
        # minNeighbors = 5,
        scaleFactor = 7,
        minNeighbors = 160,
        minSize = (50, 50),
    )

    # Draw a rectangle around the faces
    for x, y, w, h in objetos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("Se detecta correctamente", cant + 1)
        cant = cant + 1

    if anterior != len(objetos):
        anterior = len(objetos)
        log.info("faces: " + str(len(objetos)) + " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Display the resulting frame
    cv2.imshow("Video", frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
