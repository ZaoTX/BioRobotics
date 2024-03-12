import cv2
import numpy as np
from scipy import ndimage

def get_image(cap, killer):
    # read image from pi car camera
    ret, frame = cap.read()
    if frame is not None:
        frame = frame.astype("uint8")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    return np.zeros((480, 640))


def rotate_image(img, angle):
    padded_img = np.pad(img, 1, mode='constant', constant_values=255)
    # Rotate the padded image
    rotated_img = ndimage.rotate(padded_img, angle, reshape=False, mode='nearest')

    # Find bounding box of non-zero pixels in rotated image
    non_zero_indices = np.nonzero(rotated_img)
    bbox = np.min(non_zero_indices[0]), np.max(non_zero_indices[0]), np.min(non_zero_indices[1]), np.max(
        non_zero_indices[1])

    # Crop the rotated image to remove black edges
    cropped_rotated_img = rotated_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    return rotated_img


# Example usage
cap = cv2.VideoCapture(0)  # Change index if using a different camera
killer = None  # Assuming killer is defined elsewhere

while True:
    frame = get_image(cap, killer)

    # Rotate the image
    rotated_frame =rotate_image(frame,-60)
    # Display the rotated image
    cv2.imshow('Rotated Image', rotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
