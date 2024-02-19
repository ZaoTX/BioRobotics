import cv2
import picamera
from picamera.array import PiRGBArray
import numpy as np

camera = picamera.PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))

# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[(j + 1) % n][0]), (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im)
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

    image = frame.array
    qrDecoder = cv2.QRCodeDetector()
    # Detect and decode the qrcode
    data, bbox, rectifiedImage = qrDecoder.detectAndDecode(image)
    if len(data) > 0:
        print("Decoded Data : {}".format(data))
        #display(image, bbox)
        rectifiedImage = np.uint8(rectifiedImage);
        cv2.imshow("Rectified QRCode", rectifiedImage);
    else:
        print("QR Code not detected")
        cv2.imshow("Results", image)

    rawCapture.truncate(0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break