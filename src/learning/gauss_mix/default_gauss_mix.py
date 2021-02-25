import numpy as np
import cv2

img = cv2.imread('blank.png')
cap = cv2.VideoCapture("/home/xwl/PycharmProjects/gmm-sort-track/input/mot.avi")

if not cap.isOpened():
    # 如果没有检测到摄像头，报错
    raise Exception('Check if the camera is on.')

mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
frame_count = 0
while cap.isOpened():
    catch, frame = cap.read()
    frame_count += 1
    if not catch:
        print('The end of the video.')
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = mog.apply(gray).astype('uint8')
        mask = cv2.medianBlur(mask, 3)
        cv2.imwrite('./default/mask' + str(frame_count) + '.jpg', mask)
        print('writing mask' + str(frame_count) + '.jpg...')