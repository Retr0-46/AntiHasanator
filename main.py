import cv2
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))


def FindCircles(frame):

    circles = cv.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 20)
    if circles is not None:
        '''
        a, b, c = circles.shape
        print(str(circles))
        for i in range(b):
            cv.circle(frame, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(frame, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3,
                      cv.LINE_AA)  # draw center of circle'''

        #cv.imshow("detected circles", frame)
        print('Circle!')


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 90)
    # write the flipped frame
    #out.write(frame)

    #low_red = (7,40,60)
    #high_red = (18,255,200)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 85, 110], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")

    # красный в диапазоне фиолетового оттенка
    lower_violet = np.array([165, 85, 110], dtype="uint8")
    upper_violet = np.array([180, 255, 255], dtype="uint8")

    red_mask_orange = cv2.inRange(frame, lower_red, upper_red)  # применяем маску по цвету
    red_mask_violet = cv2.inRange(frame, lower_violet, upper_violet)  # для красного таких 2

    red_mask_full = red_mask_orange + red_mask_violet

    #frame = cv2.inRange(frame, red_mask_orange, red_mask_violet)

    #cv.imshow('frame', red_mask_full)

    img = red_mask_full

    hsv_min = np.array((0, 77, 17), np.uint8)
    hsv_max = np.array((208, 255, 255), np.uint8)

    thresh = cv.inRange(frame, hsv_min, hsv_max)
    _, contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        if len(cnt) > 4:
            ellipse = cv.fitEllipse(cnt)
            cv.ellipse(img, ellipse, (0, 0, 255), 2)

    cv.imshow('contours', img)

    FindCircles(red_mask_full)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
#out.release()
cv.destroyAllWindows()


#2
mport cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('o.png')
img = image.copy()

red_low = np.array([0, 10, 120])
red_high = np.array([15, 255, 255])
white_low = np.array([0,0,168])
white_high = np.array([0,0,255])
blue_low = np.array([110, 50, 50])
blue_high = np.array([130, 255, 255])

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
mask_b = cv2.inRange(hsv, red_low, red_high)
mask_r = cv2.inRange(hsv, blue_low, blue_high)
mask_w = cv2.inRange(hsv, white_low, white_high)


cv2.imshow("HSV", hsv)
cv2.imshow("Blue", mask_b)
cv2.imshow("Red", mask_r)
cv2.imshow("White", mask_w)
cv2.waitKey()

def get_cnts(mask_):
	imgray = mask_ # cv2.cvtColor(cv2.cvtColor(mask_, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(imgray, (5, 5), 0)
	edges = cv2.Canny(blur, 0, 0)
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	return cnt



# Находим контуры квадратов
contours_r = get_cnts(mask_r)
contours_b = get_cnts(mask_b)
contours_w = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

out = []

# Белый квадрат
left_up, left_down, right_down, right_up = \
contours_w[0][0], contours_w[1][0], contours_w[2][0], contours_w[3][0]
x_centre = left_up[0] + (right_up[0] - left_up[0]) / 2
y_centre = right_up[1] + (right_down[1] - right_up[1]) / 2
cv2.circle(img, (int(x_centre), int(y_centre)), radius = 0, color = (0, 0, 255), thickness = 1)
out.append((x_centre, y_centre))
cv2.rectangle(img, left_up, right_down, (0, 0, 255), 2)

# Красный круг (X)
coord_g = cv2.moments(contours_r)
cX = coord_g["m10"] / coord_g["m00"]
cY = coord_g["m01"] / coord_g["m00"]
cv2.circle(img, (round(cX), round(cY)), 1, (0, 255, 0), -1)
out.append((cX, cY))

# Синий круг (Y)
coord_g = cv2.moments(contours_b)
cX = coord_g["m10"] / coord_g["m00"]
cY = coord_g["m01"] / coord_g["m00"]
cv2.circle(img, (round(cX), round(cY)), 1, (0, 0, 255), -1)
out.append((cX, cY))

# Линия X
cv2.line(img, (int(out[1][0]), int(out[1][1])), (int(out[2][0]), int(out[2][1])), (255, 255, 255), thickness=1)

cv2.imshow("Result", img)
cv2.waitKey()


#3
import cv2
import numpy as np
image = cv2.imread('img_1.png')
img = image.copy()
h, w, channels = img.shape
half = h // 2
top = img[:half, :]
cv2.imwrite('top.jpg', top)
cv2.imshow("Red", top)
red_low = np.array([0,85,110])
red_high = np.array([10, 255, 255])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_r = cv2.inRange(hsv, red_low, red_high)
red = cv2.bitwise_and(img, img, mask=mask_r)
gray_r = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
contours_r = cv2.findContours(gray_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
tr=cv2.drawContours(img, contours_r, -1, (0,255,0), 1)
cv2.imshow("Red", tr)
cv2.waitKey()
red_shapes = []
for contour in contours_r:
    print('')
    print(contour)
    print('')
    left_up, left_down, right_down, right_up = \
        contour[0][0], contour[1][0], contour[2][0], contour[3][0]
    x_centre = left_up[0] + round((right_up[0] - left_up[0]) / 2)
    y_centre = right_up[1] + round((right_down[1] - right_up[1]) / 2)
    cv2.circle(img, (x_centre, y_centre), radius=0, color=(255, 255, 255), thickness=1)
    red_shapes.append([x_centre, y_centre])
    cv2.rectangle(img, left_up, right_down, (255, 255, 255), 2)
    print(left_up, left_down, right_down, right_up)
output = [red_shapes]
h, w, channels = img.shape
half = h // 2
top = img[:half, :]
cv2.imwrite('top.jpg', top)
cv2.imshow("Red", top)
