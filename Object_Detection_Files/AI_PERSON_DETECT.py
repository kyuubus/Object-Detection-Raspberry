import argparse
import sys
import time
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from datetime import datetime
import RPi.GPIO as GPIO
from time import sleep


import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

GPIO.setmode(GPIO.BCM)      # We are using the BCM pin numbering

Servo_pin1=17
Servo_pin2=27

GPIO.setup(Servo_pin1, GPIO.OUT)     
GPIO.setup(Servo_pin2, GPIO.OUT)

pan = GPIO.PWM(Servo_pin1, 50)
tilt = GPIO.PWM(Servo_pin2, 50)

# panAngle=0
# tiltAngle=0

pan.start(0)
tilt.start(0)

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  dispW=640
  dispH=480

  picam2=Picamera2()
  picam2.preview_configuration.main.size=(dispW,dispH)
  picam2.preview_configuration.main.format='RGB888'
  picam2.preview_configuration.align()
  picam2.configure("preview")
  picam2.start()

  # Setting jika menggunakan WebCam
  #cap = cv2.VideoCapture(camera_id)
  #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  
  boxColor=(255,0,0)
  boxWeight=2
  
  labelHeight=1.5
  labelColor=(0,255,0)
  labelWeight=(2)

  # Memulai pembacaan Object dari Camera
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Kodingan Main untuk terus memutar dan mengambil gambar
  while True:
    image=picam2.capture_array()

    #untuk memutar gambar
    #image = cv2.flip(image, 1)

    #mengubah urutan warna dari BGR menjadi RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Mengubah Hasil gambar dari rgb_image, menambahkan Image dari db tensor
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    # Menyamakan Deteksi gambar dengan db image tensor
    detection_result = detector.detect(input_tensor)
    # Menambahkan point kunci pada gambar
    #image = utils.visualize(image, detection_result)
    for detection in detection_result.detections:
        UL=(detection.bounding_box.origin_x,detection.bounding_box.origin_y)
        LR=(detection.bounding_box.origin_x+detection.bounding_box.width,detection.bounding_box.origin_y+detection.bounding_box.height)
        objName=detection.categories[0].category_name
        if objName=='person':
            image=cv2.rectangle(image,UL,LR,boxColor,boxWeight)
            cv2.putText(image,objName,UL,cv2.FONT_HERSHEY_PLAIN,labelHeight,labelColor,labelWeight)
            objname=detection.categories[0].category_name
            #print(detection.bounding_box.origin_x+detection.bounding_box.width/2-dispW/2)
            error=(detection.bounding_box.origin_x+detection.bounding_box.width/2)-dispW/2
            print(error)
#             if error>50:
#                 pan.ChangeDutyCycle(8)
#                 sleep(0.15)
#                 #pan.ChangeDutyCycle(0)
# #                 if error<90:
# #                     error=90
# #                     pan.ChangeDutyCycle(0)
#             
#             if error<50:
#                 pan.ChangeDutyCycle(6)
#                 sleep(0.15)
# #                 if error>90:
# #                     error=90
# #                     pan.ChangeDutyCycle(0)
#             if error==1:
#                 pan.ChangeDutyCycle(0)
            #picam2.capture_file(now+".png")
            
            
            
    # Perhitungan FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Menampilkan FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Memberhentikan program jika key q dipencet
    if cv2.waitKey(1) == ord('q'):
      break
    cv2.imshow('object_detector', image)
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()


