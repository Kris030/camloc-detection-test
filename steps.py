import numpy as np
import cv2

steps = [
    lambda frame, *_: (cv2.flip(frame, 1), None),
    # lambda frame, *_: (cv2.flip(cv2.flip(frame, 1), 0), None),
    # lambda frame, *_: (cv2.flip(frame, 1), None),

    # lambda frame, *_: (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lambda res: cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)),

    # lambda frame, *_: (cv2.blur(frame, (10, 10)), None),

    # lambda frame, *_: (
    #     cv2.bitwise_and(
    #         frame, frame,
    #         mask=cv2.inRange(
    #             cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
    #             np.array([100, 0, 0]),
    #             np.array([360, 360, 360]),
    #         )
    #     ),
    #     None
    # ),

    # *[lambda frame, *_: (frame, None) for _ in range(10)],
]
