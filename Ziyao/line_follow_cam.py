import cv2
from simple_pid.pid import PID
import numpy as np
from time import sleep
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
    if int(speed_left)>=4095:
        speed_left = 4094
    if int(speed_right)>=4095:
        speed_right = 4095
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
################################ Actions ################################
########### QR Code ###########
def Turn720Deg(linv_ori,angv_ori):
    #Turn the car for 720
    ang_v = 180
    # turning speed
    time_needed = (720 / ang_v) * 0.55
    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def TurnAround(linv_ori, angv_ori):
    # Turn the car for 720
    ang_v = 180
    # turning speed
    time_needed = (180 / ang_v) * 0.6
    set_car_control(linear_v=0, angular_v=ang_v)
    time.sleep(time_needed)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)
    return time_needed
def Stop10s(linv_ori, angv_ori):
    set_car_control(linear_v=0, angular_v=0)
    time.sleep(10)
    set_car_control(linear_v=linv_ori, angular_v=angv_ori)

    return 10
########### Avoid Duck ###########
def stop_car():
    set_speed(0,0)
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
    go_straight_n_seconds(0, 0, 1.2)
    time.sleep(1.2)
    # # # turn left 90 degrees
    time_needed = turn_left_90_degrees(0, 0)
    time.sleep(time_needed)
    # # # go for 1 second
    go_straight_n_seconds(0,0,1)
    time.sleep(1)
    # # turn right 90 degrees
    turn_right_60_degrees(linv_ori, angv_ori)

def analyze_image(image, prev_value):
    img_bottom = image[-100:, :]
    blur = cv2.GaussianBlur(img_bottom, (5, 5), 0)
    cv2.imwrite("last_frame_straight.png", blur)
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


def init_cam():
    cap = cv2.VideoCapture(0)

    if not (cap.isOpened()):
        raise Exception("Camera is not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap


def get_image(cap, killer):
    # read image from pi car camera
    ret, frame_ori = cap.read()
    if killer.kill_now:
        return np.zeros((480, 640))
    frame_ori = frame_ori.astype("uint8")
    frame = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2GRAY)
    # save last frame
    #cv2.imwrite("Ducks/last_frame.png", frame)
    return frame, frame_ori


def close_cam(cap):
    cap.release()


def set_car_control(linear_v, angular_v):
    # map from speed to wheel motor input
    a, b = 0.027384678763152703, -0.2914328262712497
    diff = (angular_v - b) / a
    j, k = 0.06430834737443417, 78.99787271119764
    sum = (linear_v - k) / j

    right_in = (diff + sum) / 2.
    left_in = sum - right_in

    # drive car with left and right control
    #print(left_in, right_in)
    set_speed(left_in, right_in)
    return

def detect_qrcode(image,detector): # takes RGB as input
    #barcodes = decode(image)
    #detector = cv2.QRCodeDetector()
    data, vertices_array, binary_qrcode = detector.detectAndDecodeCurved(image)
    if len(data)>0:
        return True
    # if len(barcodes) > 0:
    #     print("Decoded Data : {}".format(barcodes))
    #     if("car_rotate_720" in str(barcodes[0].data) ):
    #         # time_needed = Turn720Deg(linv_ori,angv_ori)
    #         return True,"car_rotate_720"
    #     elif("car_turn_around" in str(barcodes[0].data)):
    #         # time_needed = TurnAround(linv_ori,angv_ori)
    #         return True,"car_turn_around"
    #     elif ("car_stop_10s" in str(barcodes[0].data)):
    #         # time_needed = Stop10s(linv_ori, angv_ori)
    #         return True,"car_stop_10s"
    # else:
        #print("QR Code not detected")
    return False, 0
def qrcode_perform_action(action):
    if ("car_rotate_720" ==action):
        time_needed = Turn720Deg(0,0)
        return time_needed
    elif ("car_turn_around"  ==action):
        time_needed = TurnAround(0,0)
        return time_needed
    elif ("car_stop_10s"  ==action):
        time_needed = Stop10s(0, 0)
        return time_needed
def detect_yellow_area(image,last_duck_detected):
    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([15, 25, 25])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Check if there's yellow in the image
    num_white_pixels = np.sum(res >= 200)  #
    print (num_white_pixels)

    if last_duck_detected:
        if num_white_pixels>=1:
            return  True
    # # Print the result
    if num_white_pixels >6:
        print("Yellow detected in the image!")
        return True

    return False
def control_car(dry_run=False):
    cap = init_cam()
    killer = GracefulKiller()
    detector = cv2.QRCodeDetector()
    image_gray, image_ori = get_image(cap, killer)
    # cv2.imshow("Image ori", image_ori)
    # cv2.imshow("Image gray", image_gray)
    image_middle = int(image_gray.shape[1] / 2)
    current_position = analyze_image(image_gray, 0)
    controller = PID(1, 0.1, 0.05, setpoint=image_middle, output_limits=(0, 6.28), starting_output=3.14,
                     sample_time=1. / 30.)
    duck_detected = False
    qrcode_detected = False
    action = None
    linear_v = 300
    angular_v = 0
    last_duck_detected = False
    #last_qrcode_detected = False
    while not killer.kill_now:
        if duck_detected:
            stop_car()
            print("car stopped")
            last_duck_detected = True
            time.sleep(0.5)
        # elif qrcode_detected:
        #     time_needed = qrcode_perform_action(action)
        #     #sleep to avoid the camera capturing qr code again
        #     time.sleep(time_needed)
        #     print("perform qr code action")
        else:
            print("line following")
            if(last_duck_detected):
                print("last frame detected something")
                # do the PID analyze again
                image_gray, image_ori = get_image(cap, killer)
                current_position = analyze_image(image_gray, current_position)
                last_qrcode_detected = False
                last_duck_detected = False
            #start_time = time.time()
            angular_v = controller(current_position) - 3.14
            #current setup works
            linear_v = 300
            angular_v *=30
            # if (current_position < (image_gray.shape[1] / 5)) or (current_position > (image_gray.shape[1] - image_gray.shape[1] / 5)):
            #     linear_v = 0
            #     angular_v = angular_v * 3

            if not dry_run:
                set_car_control(linear_v, angular_v)

        image_gray,image_ori = get_image(cap, killer)
        #qrcode_detected, action = detect_qrcode(image_gray, detector)

        current_position = analyze_image(image_gray, current_position)
        image_gray, image_ori = get_image(cap, killer)
        duck_detected = detect_yellow_area(image_ori,last_duck_detected)

    set_speed(0, 0)
    print("process terminated")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dry",
                        action="store_true", default=False,
                        help="do not drive motor")
    args = parser.parse_args()
    control_car(args.dry)
