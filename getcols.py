#!/bin/env python
# 
# Calibrate the colors to track
# Use the sliders highlight the desired colors to track
# Use `q` to quit and save the calibration

import numpy as np
import signal
import cv2
import sys

save_file = sys.argv[1] if len(sys.argv) > 1 else 'colors.csv'

def save(*_):
    # Save calibration to a file
    with open(save_file, 'w') as f:
        f.write(f'{hMin},{sMin},{vMin},{hMax},{sMax},{vMax}')
    print(f'saved values to {save_file}')
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, save)

# Read config from colors.csv
try:
    with open(save_file) as f:
        cls = list(map(int, f.read().splitlines()[0].split(',')))
        HMin = phMin = cls[0]
        SMin = psMin = cls[1]
        VMin = pvMin = cls[2]
        HMax = phMax = cls[3]
        SMax = psMax = cls[4]
        VMax = pvMax = cls[5]
except:
    HMin = phMin = SMin = psMin = VMin = pvMin = 0
    HMax = phMax = SMax = psMax = VMax = pvMax = 255

def nothing(_): pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMin', 'image', HMin)
cv2.setTrackbarPos('SMin', 'image', SMin)
cv2.setTrackbarPos('VMin', 'image', VMin)
cv2.setTrackbarPos('HMax', 'image', HMax)
cv2.setTrackbarPos('SMax', 'image', SMax)
cv2.setTrackbarPos('VMax', 'image', VMax)

# Using a named pipe
# cap = cv2.VideoCapture()
# cap.open('../pipes/anyad')

# Using camera index
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(cv2.blur(frame, (10, 10)), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Print if there is a change in HSV value
    if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
        # print(f'{hMin=}; {sMin=}; {vMin=}\n{hMax=}; {sMax=}; {vMax=}')
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image', frame)

save()