#!/bin/python

from math import sqrt, ceil
from steps import steps
import numpy as np
import cv2

def main():
    WIN = 'test'

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WIN, flags=cv2.WINDOW_GUI_NORMAL)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ss = len(steps) + 1

    sw = ceil(sqrt(ss))
    sh = ceil(ss / sw)

    img = np.zeros((sh * h, sw * w, 3), np.uint8)

    while cv2.waitKey(1) != ord('q'):
        ret, frame = cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        x, y = 1, 0
        img[:h, :w] = frame
        si = 1
        for s in steps:
            xw, yh = x * w, y * h

            def do_step():
                frame, conversion = s(frame, w, h, xw, yh)

                img[
                    yh:(y + 1) * h,
                    xw:(x + 1) * w,
                ] = frame if conversion == None else conversion(frame)

                cv2.putText(
                    img,
                    str(si),
                    (xw, yh + 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (255, 255, 255),
                    5
                )

            do_step()

            si += 1
            x += 1
            if x == sw:
                x = 0
                y += 1

        cv2.imshow(WIN, img)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
