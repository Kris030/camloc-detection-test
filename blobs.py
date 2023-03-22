#!/bin/env python
# 
# Calibrate the colors to track
# Use the sliders highlight the desired colors to track
# Use `q` to quit and save the calibration

import numpy as np
import cv2

def nothing(_): pass

# Create a window
cv2.namedWindow('image')
cv2.createTrackbar('min_size', 'image', 0, 5_000, nothing)

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break
    
    filtered = cv2.inRange(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
        np.array([50, 50, 0]),
        np.array([75, 255, 255])
    )

    # find all of the connected components (white blobs in your image).
    # labels is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    blob_count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered)

    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]

    min_size = cv2.getTrackbarPos('min_size', 'image')

    im_result = np.zeros(labels.shape, np.uint8)
    for blob in range(1, blob_count):
        if sizes[blob] >= min_size:
            im_result[labels == blob + 1] = 255

    im_result = cv2.cvtColor(im_result, cv2.COLOR_GRAY2BGR)

    cv2.imshow('image', im_result)
