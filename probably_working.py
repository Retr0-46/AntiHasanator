import cv2
import numpy as np
import cv2 as cv

#cap = cv2.imread('2 круга.jpg', 1)


def FindCircles(frame):
    circles = cv.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 50, 1000)
    if len(circles) > 0:
        print(circles)


while True:
    frame1 = cv2.imread('f.jpg', 1)

    #frame = cv.flip(frame, 90)
    #frame = cv.flip(frame, 90)
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 85, 110], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")

    lower_violet = np.array([165, 85, 110], dtype="uint8")
    upper_violet = np.array([180, 255, 255], dtype="uint8")

    red_mask_orange = cv2.inRange(frame, lower_red, upper_red)
    red_mask_violet = cv2.inRange(frame, lower_violet, upper_violet)

    red_mask_full = red_mask_orange + red_mask_violet

    img = red_mask_full

    hsv_min = np.array((0, 77, 17), np.uint8)
    hsv_max = np.array((208, 255, 255), np.uint8)

    thresh = cv.inRange(frame, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        if len(cnt) > 4:
            ellipses = cv.fitEllipse(cnt)

    red_shapes = []
    try:
        for contour in contours0:
            print(contour)
            left_up, left_down, right_down, right_up = \
                contour[0][0], contour[1][0], contour[2][0], contour[3][0]
            x_centre = left_up[0] + round((right_up[0] - left_up[0]) / 2)
            y_centre = right_up[1] + round((right_down[1] - right_up[1]) / 2)
            cv2.circle(img, (x_centre, y_centre), radius=1, color=(255, 255, 255), thickness=1)
            red_shapes.append([x_centre, y_centre])
            print(left_up, left_down, right_down, right_up)
    except IndexError as E:
        print(E)
    output = [red_shapes]

    cv.imshow('contours', img)

    FindCircles(red_mask_full)
    if cv.waitKey(1) == ord('q'):
        break
    break

cv.destroyAllWindows()
