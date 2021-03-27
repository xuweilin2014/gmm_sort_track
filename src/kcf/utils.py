import cv2
import numpy as np
from table_feature import TableFeature
import fhog

# 离散傅立叶、逆变换
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!

# 实部图像
def real(img):
    return img[:, :, 0]

# 虚部图像
def imag(img):
    return img[:, :, 1]

# 两个复数，它们的积 (a + bi)(c + di) = (ac - bd) + (ad + bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res

# 两个复数，它们相除 (a + bi) / (c + di) = (ac + bd) / (c*c + d*d) + ((bc - ad) / (c*c + d*d)) * i
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0]**2 + b[:, :, 1]**2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res

# 可以将 fft 输出中的直流分量移动到频谱的中央
def rearrange(img):
    # 断言必须为真，否则会抛出异常，ndim 为数组维数
    assert(img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]

# limit 的值一定为 [0, 0, image.width, image.height]
def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    # 如果 rect[0] 也就是 x 是小于 0 的，说明 rect 图像有一部分是在 image 图像之外，那么就将 width 调整为在图像内的长度
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    # 如果 rect[1] 也就是 y 是小于 0 的，说明 rect 图像有一部分是在 image 图像之外，那么就将 height 调整为在图像内的长度
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]

    if rect[2] <= 0:
        rect[2] = 1
    if rect[3] <= 0:
        rect[3] = 1

    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert(np.all(np.array(res) >= 0))
    return res

# 经常需要空域或频域的滤波处理，在进入真正的处理程序前，需要考虑图像边界情况。
# 通常的处理方法是为图像增加一定的边缘，以适应【卷积核】在原图像边界的操作。
def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])   # modify cutWindow
    assert(cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    # 由于 roi 区域可能会超出原图像边界，因此超出边界的部分填充为原图像边界的像素
    if border != [0, 0, 0, 0]:
        # 在 OpenCV 的滤波算法中，copyMakeBorder 是一个非常重要的工具函数，它用来扩充 res 图像的边缘，将图像变大，然后以各种
        # 外插的方式自动填充图像边界，这个函数实际上调用了函数 cv2.borderInterpolate，这个函数最重要的功能就是为了处理边界
        # borderType 是扩充边缘的类型，就是外插的类型，这里使用的是 BORDER_REPLICATE，也就是复制法，也就是复制最边缘像素
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)

    return res

def extract_cn_feature(img, mapp, cell_size=1):
    height, width = img.shape[:2]
    sizeX = width // cell_size
    sizeY = height // cell_size
    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='raw_pixel', cell_size=cell_size, compressed_dim=31, table_name="CNnorm", use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        img = img[:, :, ::-1]

    h,w = img.shape[:2]
    cn_feature = cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[0][:, :, :, 0]
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    cn_feature = np.concatenate((gray, cn_feature), axis=2)

    shape = cn_feature.shape
    cn_feature.resize(shape[2], shape[0] * shape[1])

    return cn_feature
