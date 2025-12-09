#!/usr/bin/python3

import cv2
import time
import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


COUNT_LIMIT = 30
POS=(30,60)  #top-left
FONT=cv2.FONT_HERSHEY_COMPLEX #font type for text overlay
HEIGHT=1.5  #font_scale
TEXTCOLOR=(0,0,255)  #BGR- RED
BOXCOLOR=(255,0,255) #BGR- BLUE
WEIGHT=3  #font-thickness
x=0
# Grab images as numpy arrays and leave everything else to OpenCV.

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
count=0

while True:
    im = picam2.capture_array()
    cv2.putText(im,'Count:'+str(int(count)),POS,FONT,HEIGHT,TEXTCOLOR,WEIGHT)

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
        count += 1

    cv2.imshow("Camera", im)
    if count >=30:
        print(filename)
        picam2.stop()
        video_config = picam2.create_video_configuration()
        picam2.configure(video_config)

        encoder = H264Encoder(10000000)

        picam2.start_recording(encoder, "/home/pi/"+filename+".h264")
        time.sleep(5)
        picam2.stop_recording()
        count=0
        time.sleep(5)
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
    key = cv2.waitKey(100) & 0xff
