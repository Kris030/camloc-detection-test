#!/bin/python
import numpy as np
import cv2

def main():
    WIN = 'test'

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WIN, flags=cv2.WINDOW_GUI_NORMAL)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img = np.zeros((h, w * 3, 3), np.uint8)

    while cv2.waitKey(1) != ord('q'):
        ret, frame = cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        img[:h, :w] = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------

        img[:, w:w * 2] = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img[:, (w * 2):] = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        # -----------

        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # # noise removal
        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
        
        # # sure background area
        # sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)

        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)

        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1

        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0

        # markers = cv2.watershed(frame, markers)
        # frame[markers == -1] = [255, 0, 0]

        cv2.imshow(WIN, img)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
