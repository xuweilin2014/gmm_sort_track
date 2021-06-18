import numpy as np
import cv2
import fhog

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

def cut_outsize(num, limit):
    if num < 1:
        num = 1
    elif num > limit - 1:
        num = limit - 1
    return int(num)

def complexDivisionReal(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / b

    res[:, :, 0] = a[:, :, 0] * divisor
    res[:, :, 1] = a[:, :, 1] * divisor
    return res

# 实部图像
def real(img):
    return img[:, :, 0]

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


"""
尺度估计器
"""
"""
目标所在的图像块 P 的大小为 M x N，以图像块的正中间为中心，截取不同尺度的图片，这样就能够得到一系列的不同尺度的图像 Patch，形成一个图像金字塔
（搜索范围为 S 个尺度，就会有 S 张图像 Patch），针对每个图像 Patch 求其特征描述子（维度为 d 维，这里的 d 和位置估计中的维度 d 
没有任何关系），g 是高斯函数构造的输出响应大小为 1 x S，中间值最大，向两端依次减小。
"""


# noinspection PyAttributeOutsideInit
class ScaleEstimator:

    # 初始化尺度估计器
    def __init__(self, roi, image, hog):
        # dsst 算法中对一个目标，会分成 33 个尺度来进行计算
        self.n_scales = 33
        # 多尺度估计时候的尺度步长，也就是论文中的 a 的值
        self.scale_step = 1.8
        # 当前帧的尺度因子，注意当前尺度因子是累积的
        # 比如，第一帧时，物体的尺度变为初始的 1.5 倍，那么 factor = 1.5，在第二帧时，物体的尺度又变为原来第一帧的 1.2，factor = 1.5 * 1.2 = 1.8
        self.current_scale_factor = 1
        # 对新一帧图像进行检测时，公式分母中的 lambda 值
        self.scale_lambda = 0.01
        # dsst 论文中对滤波模板 H 的分子 A 和分母 B 进行在线更新时的系数
        self.scale_lr = 0.025
        # roi 为 [x, y, width, height]
        self._roi = [0, 0, 0, 0]

        self.scale_max_area = 512
        self.scale_padding = 1.0
        self.scale_sigma_factor = 0.25
        # 之前，生成的 scale_factors 为 scale_step ** [16, 15, ..., 1, 0, -1, -2, ..., -15, -16]
        # 现在，变成了 scale_step ** [16 + sd, 15 + sd, ..., 1 + sd, 0 + sd, -1 + sd, -2 + sd, ..., -15 + sd, -16 + sd]
        # 也就是倾向于得到的物体尺度更大
        self.scale_deviate = 0.5

        if hog:
            self.cell_size = 4
        else:
            self.cell_size = 1

        self._roi = roi
        # roi 是 [x, y, width, height]
        # base_width 和 base_height 为目标初始检测框的大小，current_scale_factor 就是针对这个大小来说的
        self.base_width = roi[2]
        self.base_height = roi[3]

        # guassian peak for scales (after fft)
        # 通过高斯函数构造的输出响应，大小为 (1,33)
        self.ysf = self.compute_y_label()
        self.s_hann = self.create_hanning_mats()

        # Get all scale changing rate
        # 生成一个一维的向量，shape 为 (1,33)，表示从 0 到 32 之间的数
        scale_factors = np.arange(self.n_scales)
        center = np.ceil(self.n_scales / 2.0)
        # scale_step 表示的是多尺度估计时的尺度步长，默认为 1.05
        # 生成的 scale_factors 就是尺度因子集合，将这些尺度因子和 width 以及 height 相乘，从而对原图像进行放大和缩小
        # 1.05 ** [16, 15, ..., 1, 0, -1, -2, ..., -15, -16]
        self.scale_factors = np.power(self.scale_step, center - scale_factors - 1 + self.scale_deviate)

        # get the scaling rate for compressing to the model size
        # 将图片的长和宽进行扩大和缩小
        scale_model_factor = 1.
        # 如果 base_width * base_height > scale_max_area 的话，那么算出一个因子 model_factor，
        # 用来对 base_width 和 base_height 进行调整
        if self.base_width * self.base_height > self.scale_max_area:
            scale_model_factor = (self.scale_max_area / (self.base_width * self.base_height)) ** 0.5

        self.scale_model_width = int(self.base_width * scale_model_factor)
        self.scale_model_height = int(self.base_height * scale_model_factor)

        # compute min and max scaling rate
        self.min_scale_factor = np.power(self.scale_step, np.ceil(np.log((max(5 / self.base_width, 5 / self.base_height) * (1 + self.scale_padding))) / 0.0086))

        if self.min_scale_factor > 0.5:
            self.min_scale_factor = 1e-9

        self.max_scale_factor = np.power(self.scale_step, np.floor(np.log((min(image.shape[0] / self.base_width, image.shape[1] / self.base_height) * (1 + self.scale_padding))) / 0.0086))

        # 对一维的尺度滤波器进行训练
        self.train_scale(image, True)

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

    def create_hanning_mats(self):
        _, hann_s = np.ogrid[0:0, 0:self.n_scales]
        hann_s = 0.5 * (1 - np.cos(2 * np.pi * hann_s / (self.n_scales - 1)))
        return hann_s

    # 获取尺度样本
    # 将 image 按照 n_scales = 33 个尺度因子进行缩放，并且对缩放后的每一个图片提取 fhog 特征（假设是 d 维的），那么
    # 就可以得到一个 shape 为 [d, 33, 2] 大小的矩阵，其中 2 表示矩阵的实部和虚部
    def get_scale_sample(self, image):
        xsf = None

        # n_scales 表示的就是 33 个尺度缩放
        for i in range(self.n_scales):
            # size of subwindow waiting to be detect
            # 将图像中的 patch 按照 scale_factors 中尺度因子，进行缩小和放大，具体就是对 patch 的 width 和 height 进行缩放
            # 注意在缩放时，基准是当前的物体的尺度，即 base_width/base_height * current_scale_factor，前面说过 scale_factor
            # 是当前物体尺度相对于初始物体尺度的一个比例
            patch_width = (self.base_width * self.current_scale_factor) * self.scale_factors[i]
            patch_height = (self.base_height * self.current_scale_factor) * self.scale_factors[i]

            if patch_height < 4:
                patch_height = 4
            if patch_width < 4:
                patch_width = 4

            # cx, cy 为 roi 中的中心点坐标
            cx = self._roi[0] + self._roi[2] / 2.
            cy = self._roi[1] + self._roi[3] / 2.

            if cx >= image.shape[1]:
                cx = image.shape[1] - 1
            if cy >= image.shape[0]:
                cy = image.shape[0] - 1
            if cx <= 0:
                cx = 1
            if cy <= 0:
                cy = 1

            # get the subwindow
            # 提取以点 (cx, cy) 为中心，长宽分别为 patch_width, patch_height 的图像
            im_patch = extract_image(image, cx, cy, patch_width, patch_height)

            if self.scale_model_height // self.cell_size <= 2:
                self.scale_model_height += 3 * self.cell_size
            if self.scale_model_width // self.cell_size <= 2:
                self.scale_model_width += 3 * self.cell_size

            # 下面调整 img_patch 的大小统一为 (scale_model_width, scale_model_height)，方便我们提取出相同维度的 fhog 特征
            if self.scale_model_width > im_patch.shape[1]:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 1)
            else:
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0, 3)

            mapp = {'sizeX': 0, 'sizeY': 0, 'hogFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(im_patch_resized, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)

            if i == 0:
                # 在论文中，会为每一个图像 patch 求出其特征描述子，维度为 d 维
                # 在这里也就是 d = num_features * sizeX * sizeY
                total_size = mapp['hogFeatures'] * mapp['sizeX'] * mapp['sizeY']
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

    # 检测当前图像尺度
    def detect_scale(self, image):
        # Z 是对需要进行检测 image 进行缩放，得到一个 shape 为 (total_size, n_scales, 2)
        Z = self.get_scale_sample(image)

        # compute AZ in the paper
        # 这里 Z 就是从要检测的区域提取出来的特征矩阵，A, Z 都是 (total_size, n_scales, 2) 大小的矩阵
        # A_l * Z_l，其中 A_l 和 Z_l 表示的是 1 * n_scales 大小的向量
        add_temp = cv2.reduce(cv2.mulSpectrums(Z, self.A, 0, conjB=False), 0, cv2.REDUCE_SUM)

        # compute the final y
        # 生成的 scale_response 是一个 shape 为 (1, 33) 的向量，表示对图像进行不同尺度的缩放（具体是 33 个尺度）之后，求出来的响应值
        scale_response = cv2.idft(complexDivisionReal(add_temp, (self.B + self.scale_lambda)), None, cv2.DFT_REAL_OUTPUT)

        # get the max point as the final scaling rate
        # pv:响应最大值 pi:相应最大点的索引数组
        # pi 的值为 (x, y)，其中 y = 0，x 为最大响应值在 scale_response 中的下标，比如 pi 为 (16, 0)
        _, pv, _, pi = cv2.minMaxLoc(scale_response)

        return pi

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

        # 更新 roi 框
        self.update_roi()

    # 更新尺度
    def update_roi(self):
        # 跟踪框、尺度框中心
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        # current_scale_factor 在 kcf 的 update 方法中被更新，表示当前的尺度因子
        # 使用当前尺度因子重新计算当前 roi 的 width 和 height
        self._roi[2] = self.base_width * self.current_scale_factor
        self._roi[3] = self.base_height * self.current_scale_factor

        # 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

    def get_roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = roi