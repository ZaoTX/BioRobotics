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


# controller for driving the car
def find_white_pix(line, middle_idx):
    pos = None
    index = None

    if line[middle_idx] > 0:
        pos = 'middle'
        index = middle_idx
    else:
        for i in range(middle_idx):
            try:
                if line[middle_idx + i + 1] > 0:
                    pos = 'right'
                    index = middle_idx + i + 1
                    break
            except:
                pass

            try:
                if line[middle_idx - i - 1] > 0:
                    pos = 'left'
                    index = middle_idx - i - 1
                    break
            except:
                pass

    return pos, index


def analyze_image(image, prev_value):
    img_bottom = image[-300:, :]
    blur = cv2.GaussianBlur(img_bottom, (5, 5), 0)
    ret, binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    base_line = binary_img[-1]
    middle = int(base_line.shape[0] / 2)

    root_pos, root_index = find_white_pix(base_line, middle)
    middle_pos, middle_index = find_white_pix(binary_img[-30], middle)

    current_value = 0

    if root_index is None:
        if middle_index is not None:
            current_value = middle_index
        else:
            current_value = prev_value
    elif middle_index is None:
        current_value = root_index
        current_value = image.shape[1] if binary_img[:, middle:].sum() > binary_img[:, :middle].sum() else 0
    elif abs(middle_index - middle) < abs(root_index - middle):
        current_value = middle_index
    else:
        current_value = root_index

    return current_value



def get_image(frame):
    # read image from pi car camera
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # save last frame
    cv2.imwrite("last_frame.png", frame)
    return frame



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

def control_car(dry_run=False):
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    rawCapture = picamera.array.PiRGBArray(camera, size=(640, 480))
    rawCapture.truncate(0)
    image = np.zeros((480, 640))
    camera.capture(image, 'rgb')
    image_middle = int(image.shape[1] / 2)
    controller = PID(1, 0.1, 0.05, setpoint=image_middle, output_limits=(0, 6.28), starting_output=3.14,
                     sample_time=1. / 30.)
    current_position = image_middle

    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        angular_v = 0
        linear_v = 300
        set_car_control(linear_v, angular_v)

        rawCapture.truncate(0)
        cv2.imshow("Image", image_dot)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("process terminated")
            set_speed(0, 0)
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dry",
                        action="store_true", default=False,
                        help="do not drive motor")
    args = parser.parse_args()
    control_car(args.dry)
