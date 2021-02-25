import numpy as np
import cv2
from gauss_mix import GuassMixBackgroundSubtractor

def getkpoints(imag, input1):
    mask1 = np.zeros_like(input1)
    x = 0
    y = 0
    w1, h1 = input1.shape
    input1 = input1[0:w1, 200:h1]

    try:
        w, h = imag.shape
    except:
        return None

    mask1[y:y + h, x:x + w] = 255          # 整张图片像素
    keypoints = []
    kp = cv2.goodFeaturesToTrack(input1, 200, 0.04, 7)
    if kp is not None and len(kp) > 0:
        for x, y in np.float32(kp).reshape(-1, 2):
            keypoints.append((x, y))
    return keypoints

def process(image):
    grey1 = image
    grey = cv2.equalizeHist(grey1)
    keypoints = getkpoints(grey, grey1)

    if keypoints is not None and len(keypoints) > 0:
        for i in range(len(keypoints)):
            x,y = keypoints[i]
            keypoints[i] = (x + 200, y)

    return keypoints


if __name__ == '__main__':
    cap = cv2.VideoCapture("/home/xwl/PycharmProjects/gmm-sort-track/input/p.mp4")
    frame_count = 0
    mog = GuassMixBackgroundSubtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img_copy = gray_img.copy()

        points = process(gray_img)
        for x,y in points:
            cv2.circle(gray_img, (int(x), y), 6, 255, 1)
        # gray_img[gray_img != 255] = 0

        # mask = mog.apply(gray_img_copy).astype("uint8")
        # mask = cv2.medianBlur(mask, 3)
        #
        # mask = cv2.bitwise_or(mask, gray_img)
        cv2.imshow("gray", gray_img)
        # cv2.imshow('mask', mask)
        frame_count += 1
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
