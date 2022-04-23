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
