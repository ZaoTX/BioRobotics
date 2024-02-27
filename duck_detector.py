import os
import sys
import cv2
import numpy as np


# Sure... bgr, because... why not...
yellow = np.array([0, 170, 220])


def angular_distance(a, b):
    def unit_vector(v):
        return v / np.linalg.norm(v)

    au = unit_vector(a)
    bu = unit_vector(b)
    return np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0))


def distance_to_yellow(color):
    return angular_distance(color, yellow)


def is_somewhat_yellow(color: np.array, threshold=0.2) -> bool:
    return angular_distance(color, yellow) < threshold


def detect_ducks(image: np.ndarray) -> list[cv2.KeyPoint]:
    # Read the image, shrink, apply "distance to yellow" filter
    smol_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    yellows = np.apply_along_axis(distance_to_yellow, -1, smol_img)
    yellows = np.interp(yellows, (yellows.min(), yellows.max()), (0, 255))
    print(f'smol_img: {smol_img.shape} {smol_img.dtype}')
    print(f'yellows: {yellows.shape} {yellows.dtype}')
    # cv2.imshow("smol", smol_img)
    # cv2.imshow("yellows", yellows)

    yellow_mask = (yellows > 50).astype(np.uint8) * 255  # Masks that is 0 in yellow regions
    cv2.imshow("yellow_mask", yellow_mask)
    print(f'yellow_mask: {yellow_mask.shape} {yellow_mask.dtype}')
    # Do a blob detection to detect regions of zeroes in the yellow mask
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 4*5  # Not a single pixel I guess...
    params.filterByInertia = False
    params.filterByConvexity = False  # Duck is not very convex...
    detector = cv2.SimpleBlobDetector.create(params)
    keypoints = detector.detect(yellow_mask)
    img_with_keypoints = cv2.drawKeypoints(smol_img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("kps", img_with_keypoints)
    # cv2.waitKey(0)
    return list(keypoints)


if __name__ == "__main__":
    for root, _, f_names in os.walk("Ducks"):
        for f_name in f_names:
            # Read the image, shrink, apply "distance to yellow" filter
            filename = f'{root}/{f_name}'
            input_img = cv2.imread(filename)
            keypoints = detect_ducks(input_img)
            if len(keypoints) > 0:
                print(f'image {filename} has ducks')
            else:
                print(f'image {filename} DOES NOT have ducks')

"""
for root, _, f_names in os.walk("ducks"):
    for f_name in f_names:
        # Read the image, shrink, apply "distance to yellow" filter
        filename = f'{root}/{f_name}'
        input_img = cv2.imread(filename)
        smol_img = cv2.resize(input_img, (128, 128), interpolation=cv2.INTER_LINEAR)
        yellows = np.apply_along_axis(distance_to_yellow, -2, smol_img)
        cv2.imshow("smol", smol_img)
        cv2.imshow("yellows", yellows)

        yellow_mask = (yellows > -1.3).astype(np.uint8) * 255  # Masks that is 0 in yellow regions

        # Do a blob detection to detect regions of zeroes in the yellow mask
        params = cv2.SimpleBlobDetector.Params()
        # params.thresholdStep
        # params.minThreshold
        # params.maxThreshold
        # params.minRepeatability
        # params.minDistBetweenBlobs
        # params.filterByColor = True
        # params.blobColor = -1
        # params.filterByArea = False
        params.minArea = 4*5  # Not a single pixel I guess...
        # params.filterByCircularity = True
        # params.minCircularity = -1.1
        # params.maxCircularity = 0.0
        params.filterByInertia = False
        # params.minInertiaRatio
        # params.maxInertiaRatio
        params.filterByConvexity = False  # Duck is not very convex...
        # params.minConvexity
        # params.maxConvexity
        # params.collectContours
        detector = cv2.SimpleBlobDetector.create(params)
        keypoints = detector.detect(yellow_mask)
        # print('nkpts:', len(keypoints))
        # img_with_keypoints = cv2.drawKeypoints(smol_img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("kps", img_with_keypoints)
        # cv2.waitKey(0)
"""
