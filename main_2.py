import cv2
import numpy as np
import cv2 as cv

def f(frame, gray):
    faceCascade = cv2.CascadeClassifier('ng.xml')     # Выбираем устройство видеозахвата
    while True:
        #frame = cv2.imread('img_4.png', 1)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plaques = faceCascade.detectMultiScale(gray, 1.3, 5)
        for i, (x, y, w, h) in enumerate(plaques):
            roi_color = frame[y:y + h, x:x + w]
            cv2.putText(frame, str(x) + " " + str(y) + " " + str(w) + " " + str(h), (480, 220), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255))
            r = 400.0 / roi_color.shape[1]
            dim = (400, int(roi_color.shape[0] * r))
            resized = cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)
            w_resized = resized.shape[0]
            h_resized = resized.shape[1]

            frame[100:100 + w_resized, 100:100 + h_resized] = resized  # Собираем в основную картинку
            return x, y



cap = cv.VideoCapture(0)

def FindCircles(frame):
    circles = cv.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 50, 1000)
    if len(circles) > 0:
        print(circles)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 90)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    f(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv.imshow('contours', img)

    FindCircles(red_mask_full)
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
