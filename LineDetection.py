import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def ApplyCannyEdge(image):
    # filter outliers
    blur = cv.GaussianBlur(image, (5, 5), 0)
    AdaptiveGuassian = cv.adaptiveThreshold(blur,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    edges = cv.Canny(AdaptiveGuassian, 120, 50)
    #ret, binary_img = cv.threshold(blur,180,255,cv.THRESH_BINARY)
    return edges
def linePosition(image):
    #the base line of the image
    base_line = binary_img[-1]
    #middle x coordinate of the image
    middle = int(base_line.shape[0] / 2)

def getAngleFromContour(contour, ori_img):
    rect = cv.minAreaRect(contour)
    # Get the vertices of the rotated rectangle
    box = cv.boxPoints(rect)
    box = np.int0(box)

    # Draw the rotated rectangle
    cv.drawContours(ori_img, [box], 0, (0, 0, 255), 5)
    angle = rect[2]
    width = rect[1][0]
    height = rect[1][1]
    if (width < height):
        angle = 90 - angle
    else:
        angle = -angle

    # print(angle)

    return angle


def getAngleWithDirection(angle):
    if (angle < 0):

        return "left", 90 + angle
    else:
        return "right", 90 - angle
def getAngle(dir,angle):
    if(dir == "left"):
        print(-angle)
        return -angle
    else:
        print(angle)
        return angle

def LineDetectionWithContoursAndShowImage(image, ori_img):
    contours, hierarchy = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # print(str(len(contours)))
    largest_contour = None
    dir = None
    angle = None
    centroid_x = None
    if (len(contours) > 0):
        largest_contour = max(contours, key=cv.contourArea)
        if (validateContour(largest_contour)):
            angle = getAngleFromContour(largest_contour, ori_img)
            dir, angle = getAngleWithDirection(angle)
            M = cv.moments(largest_contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            #print(dir, angle)
            # draw contours
            cv.drawContours(ori_img, [largest_contour], -1, (0, 255), 3)

    return ori_img, dir, angle,centroid_x


def validateContour(contour):
    min_contour_points = 10  # Adjust the minimum number of points as needed

    if contour is not None and len(contour) > min_contour_points:
        # Valid contour
        return True
    else:
        return False


def preprocessImage(img):
    ori_img = img
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # print("Image type:", img.dtype)
    edges = ApplyCannyEdge(img)
    img, dir, angle,centroid_x = LineDetectionWithContoursAndShowImage(edges, img)
    angle = getAngle(dir,angle)
    return img, dir, angle, centroid_x




