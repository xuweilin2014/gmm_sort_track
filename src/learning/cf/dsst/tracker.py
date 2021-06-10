import numpy as np
import cv2
import fhog

import sys

PY3 = sys.version_info >= (3,)

if PY3:
    xrange = range


# ffttools
# 离散傅里叶变换、逆变换
def fftd(img, backwards=False, byRow=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    # return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!
    # DFT_INVERSE: 用一维或二维逆变换取代默认的正向变换,
    # DFT_SCALE: 缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用。 
    # DFT_COMPLEX_OUTPUT: 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性

    if byRow:
        return cv2.dft(np.float32(img), flags=(cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT))
    else:
        return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))


# 实部图像
def real(img):
    return img[:, :, 0]


# 虚部图像
def imag(img):
    return img[:, :, 1]


# 两个复数，它们的积 (a+bi)(c+di)=(ac-bd)+(ad+bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


# 两个复数，它们相除 (a+bi)/(c+di)=(ac+bd)/(c*c+d*d) +((bc-ad)/(c*c+d*d))i
def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def complexDivisionReal(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / b

    res[:, :, 0] = a[:, :, 0] * divisor
    res[:, :, 1] = a[:, :, 1] * divisor
    return res


# 可以将 FFT 输出中的直流分量移动到频谱的中央
def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
    assert (img.ndim == 2)  # 断言，必须为真，否则抛出异常；ndim 为数组维数
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2  # shape[0] 为行，shape[1] 为列
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# recttools
# rect = {x, y, w, h}
# x 右边界
def x2(rect):
    return rect[0] + rect[2]


# y 下边界
def y2(rect):
    return rect[1] + rect[3]


# 限宽、高
def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


# 取超出来的边界
def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


# 经常需要空域或频域的滤波处理，在进入真正的处理程序前，需要考虑图像边界情况。
# 通常的处理方法是为图像增加一定的边缘，以适应 卷积核 在原图像边界的操作。
def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


def cut_outsize(num, limit):
    if num < 0:
        num = 0
    elif num > limit - 1:
        num = limit - 1
    return int(num)

# 对于一个 img, 提取出一个 patch（就是尺度变换的 33 个 patch）
# patch 宽度和高度为 patch_width 和 patch_height，并且不管尺度如何变幻，cx 和 cy 为 patch 的中心点坐标
def extract_image(img, cx, cy, patch_width, patch_height):
    # xs_s = cx - width / 2，下一步检查 xs_s 是否小于 0
    xs_s = np.floor(cx) - np.floor(patch_width / 2)
    xs_s = cut_outsize(xs_s, img.shape[1])

    # xs_e = cx + width / 2，下一步检查 xs_e 是否大于 image 的 width（注意，这里是 image 的 width，而不是 patch_wdith）
    xs_e = np.floor(cx + patch_width - 1) - np.floor(patch_width / 2)
    xs_e = cut_outsize(xs_e, img.shape[1])

    # ys_s = cy - height / 2，下一步检查 ys_s 是否小于 0
    ys_s = np.floor(cy) - np.floor(patch_height / 2)
    ys_s = cut_outsize(ys_s, img.shape[0])

    # ys_e = cy + height / 2，下一步检查 ys_e 是否大于 image 的 height
    ys_e = np.floor(cy + patch_height - 1) - np.floor(patch_height / 2)
    ys_e = cut_outsize(ys_e, img.shape[0])

    # 提取 以点 (cx, cy) 为中心，长宽分别为 patch_width, patch_height 的图像
    return img[ys_s:ys_e, xs_s:xs_e]


# noinspection PyAttributeOutsideInit
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multi_scale=False):
        # 岭回归中的 lambda 常数，正则化
        self.lambdar = 0.0001
        # 在目标框附近填充的区域大小系数
        self.padding = 2.5
        # 高斯目标的带宽
        self.output_sigma_factor = 0.125

        # 是否开启尺度自适应
        self._multiscale = multi_scale
        if multi_scale:
            # 模板大小，在计算 _tmpl_sz 时，较大边长被归一成 96，而较小边长按比例缩小
            self.template_size = 96
            self.scale_padding = 1.0
            # 多尺度估计的时候的尺度步长，也就是论文中的 a 的值
            self.scale_step = 1.05

            self.scale_sigma_factor = 0.25
            # dsst 算法中对一个目标，会分成 33 个尺度来进行计算
            self.n_scales = 33
            self.scale_lr = 0.025
            self.scale_max_area = 512
            self.scale_lambda = 0.01

            if not hog:
                print('HOG feature is forced to turn on.')

        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._hogfeatures = True if hog or multi_scale else False
        if self._hogfeatures:  # HOG feature
            # 自适应的线性插值因子
            self.interp_factor = 0.012
            # 高斯卷积核带宽
            self.sigma = 0.6
            # hog 元胞数组尺寸
            self.cell_size = 4

            print('Numba Compiler initializing, wait for a while.')

        else:  # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        self._tmpl_sz = [0, 0]
        self._roi = [0., 0., 0., 0.]
        self.size_patch = [0, 0, 0]
        self._scale = 1.
        self._alphaf = None  # numpy.ndarray (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0] * size_patch[1])
        self.hann = None  # numpy.ndarray raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0] * size_patch[1])

        # Scale properties
        self.current_scale_factor = 1
        self.base_width = 0  # initial ROI width
        self.base_height = 0  # initial ROI height
        self.scale_factors = None  # all scale changing rate, from larger to smaller with 1 to be the middle
        self.scale_model_width = 0  # the model width for scaling
        self.scale_model_height = 0  # the model height for scaling
        self.min_scale_factor = 0.  # min scaling rate
        self.max_scale_factor = 0.  # max scaling rate

        self.sf_den = None
        self.sf_num = None

        self.s_hann = None
        self.ysf = None

    ##############
    # 位置估计器 #
    #############

    # 计算一维亚像素峰值
    @staticmethod
    def subPixelPeak(left, center, right):
        divisor = 2 * center - right - left  # float
        return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

    # 初始化 hanning 窗口，函数只在第一帧被执行
    # 目的是采样时为不同的样本分配不同的权重，0.5 * 0.5 是用汉宁窗归一化[0,1]，得到矩阵的值就是每样样本的权重
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hogfeatures:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
            # 相当于把1D汉宁窗复制成多个通道
        else:
            self.hann = hann2d

        self.hann = self.hann.astype(np.float32)

    # 创建高斯峰函数，函数只在第一帧的时候执行（高斯响应）
    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    # 使用带宽 SIGMA 计算高斯卷积核以用于所有图像 X 和 Y 之间的相对位移
    # 必须都是 MxN 大小。二者必须都是周期的（即，通过一个 cos 窗口进行预处理）
    def gaussianCorrelation(self, x1, x2):
        if self._hogfeatures:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in xrange(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        else:
            # 'conjB=' is necessary! 在做乘法之前取第二个输入数组的共轭.
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                    self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                    self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    # 使用第一帧和它的跟踪框，初始化 KCF 跟踪器
    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)

        # _tmpl是截取的特征的加权平均
        self._tmpl = self.getFeatures(image, 1)
        # _prob 是初始化时的高斯响应图
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        # _alphaf 是频域中的相关滤波模板，有两个通道分别实部虚部
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)

        if self._multiscale:
            self.dsstInit(self._roi, image)

        self.train(self._tmpl, 1.0)

    # 从图像得到子窗口，通过赋值填充并检测特征
    def getFeatures(self, image, inithann, scale_adjust=1.):
        extracted_roi = [0, 0, 0, 0]
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2

        if inithann:
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if self.template_size > 1:
                # 把最大的边缩小到96，_scale是缩小比例
                # _tmpl_sz 是滤波模板的大小也是裁剪下的PATCH大小
                if padded_w >= padded_h:
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if self._hogfeatures:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (
                        2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (
                        2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        # 选取从原图中扣下的图片位置大小
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0] * self.current_scale_factor)
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1] * self.current_scale_factor)

        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        # z 是当前帧被裁剪下的搜索区域
        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:  # 缩小到96
            z = cv2.resize(z, tuple(self._tmpl_sz))

        if self._hogfeatures:
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)
            # size_patch为列表，保存裁剪下来的特征图的【长，宽，通道】
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1],
                                               self.size_patch[2])).T  # (size_patch[2], size_patch[0]*size_patch[1])

        else:  # 将 RGB 图变为单通道灰度图
            if z.ndim == 3 and z.shape[2] == 3:
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            elif z.ndim == 2:
                FeaturesMap = z

            # 从此 FeatureMap 从 -0.5 到 0.5
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            # size_patch 为列表，保存裁剪下来的特征图的【长，宽，1】
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if inithann:
            self.createHanningMats()

        FeaturesMap = self.hann * FeaturesMap  # 加汉宁（余弦）窗减少频谱泄露
        return FeaturesMap

    # 使用当前图像的检测结果进行训练
    # x是当前帧当前尺度下的特征， train_interp_factor是interp_factor
    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        # alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        # _prob是初始化时的高斯响应图，相当于y
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)

        # _tmpl是截取的特征的加权平均
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        # _alphaf是频域中相关滤波模板的加权平均
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    # 检测当前帧的目标
    # z是前一帧的训练/第一帧的初始化结果，x是当前帧当前尺度下的特征，peak_value是检测结果峰值
    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        # 得到响应图
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))

        # pv:响应最大值 pi:相应最大点的索引数组
        _, pv, _, pi = cv2.minMaxLoc(res)
        # 得到响应最大的点索引的float表示
        p = [float(pi[0]), float(pi[1])]

        # 使用幅值做差来定位峰值的位置
        if 0 < pi[0] < res.shape[1] - 1:
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if 0 < pi[1] < res.shape[0] - 1:
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        # 得出偏离采样中心的位移
        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        # 返回偏离采样中心的位移和峰值
        return p, pv

    # 基于当前帧更新目标位置
    def update(self, image):
        # 修正边界
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 1
        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 2

        # 跟踪框、尺度框中心
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        # peak_value 表示尺度不变时检测峰值结果
        # loc 表示新的最大响应值离 roi 中心的位移
        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        # 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
        # loc 是中心相对移动量
        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale * self.current_scale_factor
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale * self.current_scale_factor

        # 使用尺度估计
        if self._multiscale:
            # roi 表示的为 [x, y, width, height]
            if self._roi[0] >= image.shape[1] - 1:
                self._roi[0] = image.shape[1] - 1
            if self._roi[1] >= image.shape[0] - 1:
                self._roi[1] = image.shape[0] - 1
            if self._roi[0] + self._roi[2] <= 0:
                self._roi[0] = -self._roi[2] + 2
            if self._roi[1] + self._roi[3] <= 0:
                self._roi[1] = -self._roi[3] + 2

            # 更新尺度
            scale_pi = self.detect_scale(image)
            self.current_scale_factor = self.current_scale_factor * self.scale_factors[scale_pi[0]]
            if self.current_scale_factor < self.min_scale_factor:
                self.current_scale_factor = self.min_scale_factor

            self.train_scale(image)

        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 2

        assert (self._roi[2] > 0 and self._roi[3] > 0)

        # 使用当前的检测框来训练样本参数
        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi

    #############
    # 尺度估计器 #
    #############
    """
    目标所在的图像块 P 的大小为 M x N，以图像块的正中间为中心，截取不同尺度的图片，这样就能够得到一系列的不同尺度的图像 Patch
    （搜索范围为 S 个尺度，就会有 S 张图像 Patch），针对每个图像 Patch 求其特征描述子（维度为 d 维，这里的 d 和位置估计中的维度 d 
    没有任何关系），g 是高斯函数构造的输出响应大小为 1 x S，中间值最大，向两端依次减小。
    """

    # 通过一维的高斯函数构造的响应输出，也就是一个一维的标签，shape 为 (1, 33) (默认)
    def compute_y_label(self):
        # scale_sigma2 表示的是一维高斯函数的 sigma 值，也就是方差（注意不是标准差，而是方差）
        scale_sigma2 = (self.n_scales / self.n_scales ** 0.5 * self.scale_sigma_factor) ** 2
        # res 是一个一维的向量，表示从 [0, 32] 的数字，shape 为 (1,33)
        _, res = np.ogrid[0:0, 0:self.n_scales]
        # 这个求出的 center 其实就是一维高斯函数中的均值，这里简单的以 33 / 2 来作为均值
        center = np.ceil(self.n_scales / 2.0)
        # 使用高斯函数求出最后的响应值标签，也是一个一维向量，shape 为 (1,33)，中间值最大为 1，向两端依次减小
        res = np.exp(-0.5 * (np.power(res + 1 - center, 2)) / scale_sigma2)
        # 对响应结果进行 fft 变换
        return fftd(res)

    def createHanningMatsForScale(self):
        _, hann_s = np.ogrid[0:0, 0:self.n_scales]
        hann_s = 0.5 * (1 - np.cos(2 * np.pi * hann_s / (self.n_scales - 1)))
        return hann_s

    # 初始化 dsst 中的尺度估计器
    def dsstInit(self, roi, image):
        # roi 是 [x, y, width, height]
        self.base_width = roi[2]
        self.base_height = roi[3]

        # guassian peak for scales (after fft)
        # 通过高斯函数构造的输出响应，大小为 (1,33)
        self.ysf = self.compute_y_label()
        self.s_hann = self.createHanningMatsForScale()

        # Get all scale changing rate
        # 生成一个一维的向量，shape 为 (1,33)，表示从 0 到 32 之间的数
        scale_factors = np.arange(self.n_scales)
        center = np.ceil(self.n_scales / 2.0)
        # scale_step 表示的是多尺度估计时的尺度步长，默认为 1.05
        # 生成的 scale_factors 就是尺度因子集合，将这些尺度因子和 width 以及 height 相乘，从而对原图像进行放大和缩小
        self.scale_factors = np.power(self.scale_step, center - scale_factors - 1)

        # get the scaling rate for compressing to the model size
        # 将图片的长和宽进行扩大和缩小
        scale_model_factor = 1.
        if self.base_width * self.base_height > self.scale_max_area:
            scale_model_factor = (self.scale_max_area / (self.base_width * self.base_height)) ** 0.5

        self.scale_model_width = int(self.base_width * scale_model_factor)
        self.scale_model_height = int(self.base_height * scale_model_factor)

        # compute min and max scaling rate
        self.min_scale_factor = np.power(self.scale_step, np.ceil(np.log((max(5 / self.base_width, 5 / self.base_height) * (1 + self.scale_padding))) / 0.0086))
        self.max_scale_factor = np.power(self.scale_step, np.floor(np.log((min(image.shape[0] / self.base_width, image.shape[1] / self.base_height) * ( 1 + self.scale_padding))) / 0.0086))

        self.train_scale(image, True)

    # 获取尺度样本
    def get_scale_sample(self, image):
        xsf = None

        # n_scales 表示的就是 33 个尺度缩放
        for i in range(self.n_scales):
            # size of subwindow waiting to be detect
            # 将图像中的 patch 按照 scale_factors 中尺度因子，进行缩小和放大，具体就是对 patch 的 width 和 height 进行缩放
            patch_width = self.base_width * self.scale_factors[i] * self.current_scale_factor
            patch_height = self.base_height * self.scale_factors[i] * self.current_scale_factor

            # cx, cy 为 roi 中的中心点坐标
            cx = self._roi[0] + self._roi[2] / 2.
            cy = self._roi[1] + self._roi[3] / 2.

            # get the subwindow
            # 提取 以点 (cx, cy) 为中心，长宽分别为 patch_width, patch_height 的图像
            im_patch = extract_image(image, cx, cy, patch_width, patch_height)
            # 下面调整 img_patch 的大小统一为 (scale_model_width, scale_model_height)，方便我们提取出相同维度的 fhog 特征
            if self.scale_model_width > im_patch.shape[1]:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 1)
            else:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 3)

            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(im_patch_resized, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)

            if i == 0:
                # 在论文中，会为每一个图像 patch 求出其特征描述子，维度为 d 维
                # 在这里也就是 d = num_features * sizeX * sizeY
                total_size = mapp['numFeatures'] * mapp['sizeX'] * mapp['sizeY']
                xsf = np.zeros((total_size, self.n_scales))

            # multiply the FHOG results by hanning window and copy to the output
            # mapp['map'] 原本为 (sizeX, sizeY, num_features)，将其转变为一个列向量，(total_size, 1)
            FeaturesMap = mapp['map'].reshape((total_size, 1))
            # 加汉宁窗防止频谱泄漏
            FeaturesMap = self.s_hann[0][i] * FeaturesMap
            # xsf 是一个 (total_size, n_scales) 的矩阵，列向量为从 patch 块中提取出来的 fhog 特征
            xsf[:, i] = FeaturesMap[:, 0]

        # 对 xsf 矩阵进行 dft，也就是离散傅里叶变换
        return fftd(xsf, False, True)

    # 训练尺度估计器
    def train_scale(self, image, ini=False):
        # xsf 是一个 (total_size, n_scales, 2) 的矩阵
        # 列向量为从 patch 块中提取出来的 fhog 特征，最后的 2 表示为通过傅里叶变换得到的实部以及虚部
        xsf = self.get_scale_sample(image)

        # adjust ysf to the same size as xsf in the first time
        if ini:
            # total_size 为 sizeX * sizeY * num_features
            total_size = xsf.shape[0]
            # ysf 是一个 shape 为 (total_size, n_scales, 2) 的矩阵，最后的 2 表示也为经过傅里叶变换得到的复数的实部和虚部
            self.ysf = cv2.repeat(self.ysf, total_size, 1)

        # get new GF in the paper (delta A)
        # cv2.mulSpectrums(A, B, 0, conjB=True) 对 A 和 B 矩阵中的复数分别进行点对点相乘，并且 conjB=True 表示在相乘之前，会把 B 矩阵先进行共轭
        # 在这里就是先对 xsf 进行共轭，再把 ysf 和共轭后的 xsf 矩阵相乘，也就是对 xsf, ysf 中 (total_size, n_scales) 个复数进行对应相乘
        new_A = cv2.mulSpectrums(self.ysf, xsf, 0, conjB=True)

        # F_l * F_l_conj, 这里的 F_l 表示 xsf 中的第 l 维特征（1 <= l <= total_size），F_l 为一个 1 * n_scales 的向量
        new_B = cv2.mulSpectrums(xsf, xsf, 0, conjB=True)
        # sum(F_l * F_l_conj)（1 <= l <= total_size）
        # A = real(new_sf_den) 求出 new_sf_den 矩阵的实部，为 (total_size, n_scales)，接下来，reduce(A, 0, cv2.REDUCE_SUM)
        # 将 A 矩阵变为 (1, n_scales)，也就是将每一列上的值相加，将原矩阵压缩成一行
        new_B = cv2.reduce(real(new_B), 0, cv2.REDUCE_SUM)

        # 如果是第一次进行训练的话
        if ini:
            self.A = new_A
            self.B = new_B
        else:
            # get new A and new B
            # 对 A 和 B 进行在线更新
            self.A = cv2.addWeighted(self.A, (1 - self.scale_lr), new_A, self.scale_lr, 0)
            self.B = cv2.addWeighted(self.B, (1 - self.scale_lr), new_B, self.scale_lr, 0)

        self.update_roi()

    # 检测当前图像尺度
    def detect_scale(self, image):
        Z = self.get_scale_sample(image)

        # compute AZ in the paper
        # 这里 Z 就是从要检测的区域提取出来的特征矩阵，A, Z 都是 (total_size, n_scales, 2) 大小的矩阵
        # A_l * Z_l，其中 A_l 和 Z_l 表示的是 1 * n_scales 大小的向量
        add_temp = cv2.reduce(cv2.mulSpectrums(Z, self.A, 0, conjB=False), 0, cv2.REDUCE_SUM)

        # compute the final y
        scale_response = cv2.idft(complexDivisionReal(add_temp, (self.B + self.scale_lambda)), None, cv2.DFT_REAL_OUTPUT)

        # get the max point as the final scaling rate
        # pv:响应最大值 pi:相应最大点的索引数组
        _, pv, _, pi = cv2.minMaxLoc(scale_response)

        return pi

    # 更新尺度
    def update_roi(self):
        # 跟踪框、尺度框中心
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        # Recompute the ROI left-upper point and size
        self._roi[2] = self.base_width * self.current_scale_factor
        self._roi[3] = self.base_height * self.current_scale_factor

        # 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0
