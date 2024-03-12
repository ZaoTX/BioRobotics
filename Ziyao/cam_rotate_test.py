import cv2
import numpy as np


def get_image(cap, killer):
    # read image from pi car camera
    ret, frame = cap.read()
    if frame is not None:
        frame = frame.astype("uint8")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    return np.zeros((480, 640))


def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    # Rotate the image with padding
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_image


# Example usage
cap = cv2.VideoCapture(0)  # Change index if using a different camera
killer = None  # Assuming killer is defined elsewhere

while True:
    frame = get_image(cap, killer)

    # Rotate the image
    rotated_frame = rotate_image(frame, -45)  # Rotate counterclockwise by 45 degrees

    # Display the rotated image
    cv2.imshow('Rotated Image', rotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
