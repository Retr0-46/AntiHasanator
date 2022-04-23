import cv2
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)

def FindCircles(frame):
    circles = cv.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1000, 10000) # функция возращает координаты центра и радиус
    if len(circles) > 0:
        print(circles)
    return circles
def get_cnts(mask_):
    imgray = mask_ # cv2.cvtColor(cv2.cvtColor(mask_, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(imgray, (5, 5), 0)
    edges = cv2.Canny(blur, 0, 0)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    return cnt


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 90)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 10, 120], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")
    mask_r = cv2.inRange(frame,lower_red, upper_red)

    red_mask_full =mask_r

    img = red_mask_full

    hsv_min = np.array((0, 77, 17), np.uint8)
    hsv_max = np.array((208, 255, 255), np.uint8)

    thresh = cv.inRange(frame, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        if len(cnt) > 4:
            ellipses = cv.fitEllipse(cnt)
    red = cv2.bitwise_and(frame, frame, mask=mask_r)
    gray_r = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    contours_r= cv.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1000, 10000)
    tr = cv2.drawContours(img, contours_r, -1, (0, 255, 0), 1)
    cv2.imshow("Red", tr)

    #cv.imshow('contours', img)

    FindCircles(red_mask_full)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
