import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from numba import jit
from gauss_mix import GuassMixBackgroundSubtractor


# function to calculate gaussian
def gaussian(sigma, x):
    a = 1 / (np.sqrt(2 * np.pi) * sigma)
    b = math.exp(-(x ** 2) / (2 * (sigma ** 2)))
    return a * b


# kernel [-1,0,1]
def gaussian_kernel(sigma):
    a = gaussian(sigma, -1)
    b = gaussian(sigma, 0)
    c = gaussian(sigma, 1)
    sum = a + b + c
    if sum != 0:
        a = a / sum
        b = b / sum
        c = c / sum
    return np.reshape(np.asarray([a, b, c]), (1, 3))


# susam mask of 37 pixels
def susan_mask():
    mask = np.ones((7, 7))
    mask[0, 0] = 0
    mask[0, 1] = 0
    mask[0, 5] = 0
    mask[0, 6] = 0
    mask[1, 0] = 0
    mask[1, 6] = 0
    mask[5, 0] = 0
    mask[5, 6] = 0
    mask[6, 0] = 0
    mask[6, 1] = 0
    mask[6, 5] = 0
    mask[6, 6] = 0
    return mask


def create10by10Mask():
    arr = np.array(
        [[255, 0, 0, 0, 0, 0, 0, 0, 0, 0], [255, 255, 0, 0, 0, 0, 0, 0, 0, 0], [255, 255, 255, 0, 0, 0, 0, 0, 0, 0],
         [255, 255, 255, 255, 0, 0, 0, 0, 0, 0], [255, 255, 255, 255, 255, 0, 0, 0, 0, 0],
         [255, 255, 255, 255, 255, 0, 0, 0, 0, 0], [255, 255, 255, 255, 0, 0, 0, 0, 0, 0],
         [255, 255, 255, 0, 0, 0, 0, 0, 0, 0], [255, 255, 0, 0, 0, 0, 0, 0, 0, 0], [255, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return arr


def denoising_img(image):
    output = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            output[i, j] = np.median(
                [output[i - 1][j], output[i + 1][j], output[i][j - 1], output[i][j + 1], output[i - 1][j - 1],
                 output[i + 1][j + 1], output[i + 1][j - 1], output[i - 1][j + 1]])

    return output


def smoothing(image):
    G = gaussian_kernel(0.5)
    I = cv2.filter2D(image, -1, G + np.transpose(G))

    return I


def normalization(image):
    output = image.copy()
    output = output * (np.max(output) - np.min(output)) / 255
    return output


def plot_image(image, title):
    plt.figure()

    plt.title(title)
    plt.imshow(image, cmap='gray')

    plt.show()


def plot_multipleImage(img1, title1, img2, title2, img3, title3, img4, title4):
    plt.subplot(221)
    plt.imshow(img1, cmap=cm.gray)
    plt.title(title1)

    plt.subplot(222)
    plt.imshow(img2, cmap=cm.gray)
    plt.title(title2)

    plt.subplot(223)
    plt.imshow(img3, cmap=cm.gray)
    plt.title(title3)

    plt.subplot(224)
    plt.imshow(img4, cmap=cm.gray)
    plt.title(title4)
    plt.show()

@jit
def susan_corner_detection(img):
    img = img.astype(np.float64)
    g = 37 / 2
    circularMask = susan_mask()
    output = np.zeros(img.shape)

    for i in range(3, img.shape[0] - 3):
        for j in range(3, img.shape[1] - 3):
            ir = np.array(img[i - 3:i + 4, j - 3:j + 4])
            ir = ir[circularMask == 1]
            ir0 = img[i, j]
            a = np.sum(np.exp(-((ir - ir0) / 10) ** 6))
            if a <= g:
                a = g - a
            else:
                a = 0
            output[i, j] = a
    return output


if __name__ == '__main__':
    cap = cv2.VideoCapture("/home/xwl/PycharmProjects/gmm-sort-track/input/mot.avi")
    frame_count = 0
    mog = GuassMixBackgroundSubtractor()

    while cap.isOpened():
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img_copy = gray_img.copy()

        corners = susan_corner_detection(gray_img)
        gray_img[corners != 0] = 255
        gray_img[corners == 0] = 0

        mask = mog.apply(gray_img_copy).astype('uint8')
        mask = cv2.medianBlur(mask, 3)

        mask = cv2.bitwise_or(gray_img, mask)
        cv2.imwrite('mask' + str(frame_count) + '.jpg', mask)
        frame_count += 1
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img = cv2.imread("house.jpg", 0)
# output1 = susan_corner_detection(img)
# finaloutput1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# finaloutput1[output1 != 0] = [255, 0, 0]
# plot_image(finaloutput1, "Output Part1")  # good success

