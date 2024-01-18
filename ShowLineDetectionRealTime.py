from picamera.array import PiRGBArray
import numpy as np
import cv2
import picamera
import LineDetection

camera = picamera.PiCamera()
camera.resolution = (640,480)
rawCapture = picamera.array.PiRGBArray(camera,size=(640,480))

for frame in camera.capture_continuous(rawCapture,format="rgb",use_video_port=True):
    image = frame.array
    img_bottom = image[-300:,:]
    img,contour = LineDetection.preprocessImage(img_bottom)
    #LineDetection.LineInterpretation(contour)
    cv2.imshow("Image with line detection",img)
    rawCapture .truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
