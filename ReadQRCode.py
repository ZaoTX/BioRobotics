import math

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
    cv2.imwrite("last_frame.png", frame)
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
def init_cam():
    cap = cv2.VideoCapture(0)

    if not (cap.isOpened()):
        raise Exception("Camera is not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap
def Turn720Deg(linv_ori,angv_ori):
    #Turn the car for 720
    ang_v = 5 # in radians

    #turning speed
    speed_actual = ang_v*180/math.pi
    time_needed = 720/speed_actual

    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def TurnAround(linv_ori, angv_ori):
    # Turn the car for 720
    ang_v = 5  # in radians

    # turning speed
    speed_actual = ang_v * 180 / math.pi
    time_needed = 180 / speed_actual

    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def Stop10s(linv_ori, angv_ori):
    set_car_control(linear_v=0, angular_v=0)
    time.sleep(10)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)

    return 10
def analyse_image(image):
    GRAY_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(GRAY_image)
    if len(barcodes) > 0:
        print("Decoded Data : {}".format(barcodes))
        if("car_rotate_720" in str(barcodes[0].data) ):
            time_needed = Turn720Deg(0,0)
            return True,time_needed
        elif("car_turn_around" in str(barcodes[0].data)):
            time_needed = TurnAround(0,0)
            return True,time_needed
        elif ("car_stop_10s" in str(barcodes[0].data)):
            time_needed = Stop10s(0, 0)
            return True,time_needed
    else:
        print("QR Code not detected")
        return False, 0
def control_car(dry_run=False):
    cap = init_cam()
    killer = GracefulKiller()
    image_ori,image = get_image(cap, killer)
    qrcode_detected,time_needed=  analyse_image(image_ori)

    last_detection_time = 0
    while not killer.kill_now:
        if(not qrcode_detected):
            image_ori,image  = get_image(cap, killer)
            qrcode_detected,time_needed= analyse_image(image_ori)
            print("time_needed time =  " + str(time_needed))
            if qrcode_detected:
                last_detection_time = time.time()
                print("current time =  " + str(last_detection_time))
            print("current time =  " + str(time.time() - last_detection_time))
            if qrcode_detected and time.time() - last_detection_time >= time_needed:
                print("cam refresh")
                cap.release()  # Release the camera capture
                cap = init_cam()  # Reinitialize the camera capture
                qrcode_detected = False  # Reset the flag
                last_detection_time = 0  # Reset the last detection time
            else:
                break

def close_cam(cap):
    cap.release()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dry",
                        action="store_true", default=False,
                        help="do not drive motor")
    args = parser.parse_args()
    control_car(args.dry)

