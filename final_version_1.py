import numpy as np
import cv2

#then input picture and change color space

frame1 = cv2.imread('0.png', 1)
median = cv2.medianBlur(frame1,5)
img3=frame1.copy()
img4=frame1.copy()
img_hsv = frame1.copy()

frame = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
#here I screwed up with color ranges but this red mask probably working

hsv_min = np.array((0, 80, 80), np.uint8)
hsv_max = np.array((10, 255, 255), np.uint8)

# upper boundary RED color range values; Hue (160 - 180)
thresh = cv2.inRange(frame, hsv_min,hsv_max)
img =cv2.inRange(frame, hsv_min,hsv_max)

contours_r, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#this thing output all red centroids
red_shapes = []

try:
    for contour in contours_r:
        left_up, left_down, right_down, right_up = contour[0][0], contour[1][0], contour[2][0], contour[3][0]
        x_centre = left_up[0] + round((right_up[0] - left_up[0]) / 2)
        y_centre = right_up[1] + round((right_down[1] - right_up[1]) / 2)
        red_shapes.append([x_centre, y_centre])
        print(left_up, left_down, right_down, right_up)
except:
    pass

cv2.imshow('Blurred picture', median)
cv2.waitKey(0)

cv2.imshow('contours', img)
cv2.waitKey(0)

output = [red_shapes]

try:
    #сортирую по игрику и удаляю координаты светофора
    u=sorted(red_shapes,key=lambda y: y[1])
    print('red obj'+u)

    u1=[]
    u1.extend(u)
    u1.pop(0)
    print(u1)

    # провожу линию между фарами чтобы в дальшейшем рассматривать их как систему связанных тел
    cv2.line(img3, u1[0], u1[1], (255, 0, 0), thickness=3)
except:
    print('Red objects are not detected or theres 2 of them or less')
    pass

cv2.imshow('line', img3)
cv2.waitKey(0)

#дальше идет алгоритм по определению номера
gray = cv2.cvtColor(img3, 0)

plates_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')
plates = plates_cascade.detectMultiScale(gray, 1.2, 4)

try:
    for (x, y, w, h) in plates:
        print('n', x,y)

        plates_rec = cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
except:
    pass

cv2.imshow('img', img3)
cv2.waitKey(0)
if len(plates)>0:
    t='True'
print('Number of detected licence plates:', len (plates))
#пускай стоп линия будет голубой
blue_low = np.array([110, 50, 50])
blue_high = np.array([130, 255, 255])

hsv = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
mask_b = cv2.inRange(hsv, blue_low, blue_high)

contours_b = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
blue_shapes = []
try:
    for contour in contours_b:
        left_up, left_down, right_down, right_up = \
        contour[0][0], contour[1][0], contour[2][0], contour[3][0]
        x_centre = left_up[0] + round((right_up[0] - left_up[0]) / 2)
        y_centre = right_up[1] + round((right_down[1] - right_up[1]) / 2)
        blue_shapes.append([x_centre, y_centre])
        # print(left_up, left_down, right_down, right_up)
except:
    pass

u3=sorted(blue_shapes,key=lambda y: y[1])
print(u3)

if (len(u3))==2 and (len(u1))>=2 and t=='True':
    print('Нарушения нет')
elif (len(u3)==1 and (len(u1))>=2 and t=='True'):
    print('Нарушение')
else:
    print('Затрудняюсь ответить')
