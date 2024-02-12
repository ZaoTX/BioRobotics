from picamera.array import PiRGBArray
import numpy as np
import cv2
import picamera
import LineDetection
from simple_pid.pid import PID
# Car control
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


def set_car_control(linear_v, angular_v):
    # map from speed to wheel motor input
    a, b = 0.008603150562323695, -0.2914328262712497
    diff = (angular_v - b) / a
    j, k = 0.06430834737443417, 78.99787271119764
    sum = (linear_v - k) / j

    left_in = (diff + sum) / 2.
    right_in = sum - left_in

    # drive car with left and right control
    print(left_in, right_in)
    set_speed(left_in, right_in)

    return
def findLine(baseline, middile_index):
    linePos = 0
    for i in range(middile_index):
        if(baseline[middile_index+i]>0):
            linePos = middile_index+i+1
            break
        if(baseline[middile_index-i]>0):
            linePos = middile_index-i+1
            break
    return linePos

camera = picamera.PiCamera()
camera.resolution = (640, 480)
rawCapture = picamera.array.PiRGBArray(camera, size=(640, 480))
last_dir = None
last_angle = None
# pid controller over the angle
#controller = PID(1, 0.1, 0.05, setpoint=0, output_limits=(-3.14, 3.14), starting_output=3.14, sample_time=1. / 30.)
controller = PID(1, 0.1, 0.05, setpoint=320, starting_output=3.14,output_limits=(0, 6.28)) # 320 is the mid point of the image in x direction
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    image = frame.array
    img_bottom = image[-400:, :]

    img, dir, angle,centroid_x = LineDetection.preprocessImage(img_bottom)
    if (dir != None):
        angular_v = controller(angle) - 3.14
        # linear_v = 400 - abs(angular_v * 100 / 3.14)
        linear_v = 300
        angular_v = angular_v * 30  # remap to (-100, 100), left positive, right negative
        if (angle > 40 or angle < -40):
            print("angle more than 60")
            angular_v = angular_v * 3
        set_car_control(linear_v, angular_v)
    else:
        set_car_control(0, 0)
    cv2.imshow("Image with line detection", img)
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        set_speed(0, 0)
        break
