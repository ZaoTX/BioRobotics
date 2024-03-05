import math
from picamera.array import PiRGBArray
import picamera
import cv2
from simple_pid.pid import PID
import numpy as np
import time
import signal
from pyzbar.pyzbar import decode
import Adafruit_PCA9685
import RPi.GPIO as GPIO

pwm = Adafruit_PCA9685.PCA9685()
# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)
GPIO.setmode(GPIO.BCM)  # GPIO number  in BCM mode
GPIO.setwarnings(False)
# IN1 IN2 control the motor direction on K1/K2 port in L293 Motor board, please connect left motor to K1 or K2
# When IN1 is set to HIGH voltage  1, IN2 is LOW voltage 0, then left motor(K1 or K2) rotate backward
# When IN2 is set to HIGH voltage  1, IN1 is LOW voltage 0, then left motor(K1 or K2) rotate forward
IN1 = 23  # IN1 connect to GPIO#23(physical location# 16)
IN2 = 24  # IN2 connect to GPIO#24(physical location# 18)

# IN3 IN4 control the motor direction on K3/K4 port in L293 Motor board, please connect right motor to K3 or K4
# When IN3 is set to HIGH voltage  1, IN4 is LOW voltage 0, then right motor(K3 or K4) rotate backward
# When IN4 is set to HIGH voltage  1, IN3 is LOW voltage 0, then right motor(K3 or K4) rotate forward
IN3 = 27  # IN3 connect to GPIO#27(physical location# 13)
IN4 = 22  # IN4 connect to GPIO#22(physical location# 15)

# The motor Speed is controlled by PCA9685 PWM signal generator
# K1/K2 motor speed is controlled by PWM signal value of ENA
# K3/K4 motor speed is controlled by PWM signal value of ENB
ENA = 0  # Left motor(K1/K2) speed PCA9685 port 0
ENB = 1  # Right motor(K3/K4) speed PCA9685 port 1

# Define motor control  pins as output
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)


# direct take the code for car control
def set_speed(speed_left, speed_right):
    # make all motors moving forward at the speed of variable move_speed
    if speed_left < 0:
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN1, GPIO.HIGH)
        speed_left = -speed_left
    else:
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN1, GPIO.LOW)

    if speed_right < 0:
        GPIO.output(IN4, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        speed_right = -speed_right
    else:
        GPIO.output(IN4, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)

    pwm.set_pwm(ENA, 0, int(speed_left))
    pwm.set_pwm(ENB, 0, int(speed_right))



class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def get_image(cap, killer):
    # read image from pi car camera
    ret, frame_ori = cap.read()
    if killer.kill_now:
        return np.zeros((480, 640))
    frame = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2GRAY)
    # save last frame
    cv2.imwrite("Ducks/last_frame.png", frame_ori)
    return frame_ori,frame
def set_car_control(linear_v, angular_v):
    # map from speed to wheel motor input
    a, b = 0.0008603150562323695, -0.2914328262712497
    diff = (angular_v - b) / a
    j, k = 0.06430834737443417, 78.99787271119764
    sum = (linear_v - k) / j

    left_in = (diff + sum) / 2.
    right_in = sum - left_in

    # drive car with left and right control
    print(left_in, right_in)
    set_speed(left_in, right_in)

    return
def go_straight_n_seconds(linv_ori, angv_ori,n):
    set_car_control(linear_v=376, angular_v= 0)
    time.sleep(n)
    set_car_control(linv_ori, angv_ori)
def turn_right_90_degrees(linv_ori, angv_ori):
    # n <0 left n>0 right
    ang_v = 90
    # turning speed
    time_needed = (90 / ang_v)/3.14
    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def turn_right_60_degrees(linv_ori, angv_ori):
    # n <0 left n>0 right
    ang_v = 90
    # turning speed
    time_needed = (60 / ang_v)/3.14
    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def turn_left_90_degrees(linv_ori, angv_ori):
    # n <0 left n>0 right
    ang_v = -90
    # turning speed
    time_needed = (-110 / ang_v)/3.14
    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def avoid_duck(linv_ori, angv_ori):
    # turn right 90 degrees
    time_needed = turn_right_90_degrees(0,0)
    time.sleep(time_needed)
    # # go for 1 second
    go_straight_n_seconds(0,0,0.6)
    time.sleep(0.6)
    # # turn left 90 degrees
    time_needed = turn_left_90_degrees(0,0)
    time.sleep(time_needed)
    # # # go for 1 second
    go_straight_n_seconds(0, 0, 1.1)
    time.sleep(1.1)
    # # # turn left 90 degrees
    time_needed = turn_left_90_degrees(0, 0)
    time.sleep(time_needed)
    # time.sleep(time_needed)
    # # # go for 1 second
    # go_straight_n_seconds(0,0,1)
    # time.sleep(1)
    # # # turn right 90 degrees
    # turn_right_90_degrees(linv_ori, angv_ori)
def detect_yellow_area(image):
    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Check if there's yellow in the image
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 4*5 # Not a single pixel I guess...
    params.filterByInertia = False
    params.filterByConvexity = False  # Duck is not very convex...
    detector = cv2.SimpleBlobDetector.create(params)
    keypoints = detector.detect(res)
    # Print the result
    if len(keypoints) > 0:
        print("Yellow detected in the image!")
        return True
    else:
        print("No yellow detected in the image.")
    return False
def control_car(dry_run=False):
    killer = GracefulKiller()
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    rawCapture = picamera.array.PiRGBArray(camera, size=(640, 480))
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        if not killer.kill_now:
            image_ori = frame.array # rgb image
            image_bgr =cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", image_bgr)
            duck_detected = detect_yellow_area(image_ori)
            if duck_detected:
                # Pause the camera capture
                avoid_duck(0,0)
                time.sleep(5)
                print("Camera paused for" + str(5))
            rawCapture.truncate(0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                set_speed(0,0)
                break



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dry",
                        action="store_true", default=False,
                        help="do not drive motor")
    args = parser.parse_args()
    control_car(args.dry)

