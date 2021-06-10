from src.kcf import fhog
from utils import *
import numpy as np
import cv2

# KCF tracker
# 计算一维亚像素的峰值
def subPixelPeak(left, center, right):
    divisor = 2 * center - right - left  # float
    return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor


'''
kcf1 tracker 跟踪算法主要使用了三个公式：核回归训练提速、核回归检测提速、核相关矩阵的计算提速
1.对于读取到的第一帧，调用 tracker#init 方法，在这个方法中主要做目标跟踪的初始化工作，包括：从目标图像中获取到 fhog 特征矩阵、制作标签（也就是高斯响应图，只在第一帧的时候执行）、
初始化线性回归系数 alpha，接着调用 tracker#train 方法，train 方法主要所作的工作是求解新的线性回归参数 alpha，然后在线更新模板以及核线性回归参数 alpha
2.对于接下来读取到的每一帧，都调用 tracker#update 方法，在 update 方法中，通过 detect 方法使用快速检测公式，检测到新的最大响应值及其位置，算出新的位置距离
上一帧目标框中心的偏移量，然后移动目标框，使得新的最大响应值的位置在目标框中心。然后再调用 tracker#train 方法，根据新读入的图像，更新模板以及核线性回归参数 alpha
'''
class KCFTracker:
    def __init__(self, cn=False, hog=False, fixed_window=True, multiscale=False, peak_threshold=0.4):
        # 岭回归中的 lambda 常数，正则化
        self.lambdar = 0.0001   # regularization
        # extra area surrounding the target
        # 在目标框附近填充的区域大小系数
        self.padding = 2.5
        # bandwidth of gaussian target
        self.output_sigma_factor = 0.125
        self.peak_threshold = peak_threshold

        # 是否使用 fhog 特征
        self._hog_feature = hog
        # 是否使用 raw_pixel 特征
        self._cn_feature = cn

        if hog or cn:
            # HOG feature, linear interpolation factor for adaptation
            # 用于在 train 方法中进行训练时，对相关系数 alpha 和 template 进行在线更新
            self.interp_factor = 0.012
            # gaussian kernel bandwidth
            # 高斯卷积核的带宽
            self.sigma = 0.6
            # hog 元胞数组尺寸
            # Hog cell size
            self.cell_size = 4
        # raw gray-scale image
        # aka CSK tracker
        else:
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1

        if multiscale:
            # 模板大小，在计算 _tmpl_sz 时，较大边长被归一成 96，而较小的边按比例缩小
            self.template_size = 96   # template size
            # 多尺度估计🥌时的尺度步长
            # scale step for multi-scale estimation
            self.scale_step = 1.05
            # to downweight detection scores of other scales for added stability
            # 对于其它尺度的响应值，都会乘以 0.96，也就是乘以一个惩罚系数
            self.scale_weight = 0.96
        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.   # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])

    # 使用第一帧和它的跟踪框，初始化 KCF 跟踪器
    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        # _tmpl 是从目标图像中所获取到的 fhog 特征矩阵，shape 为 [sizeX, sizeY, 特征维数]
        # 当特征为 cn 特征的话，那么就为 11 维
        # 当特征为 fhog 特征的话，那么就为 31 维
        # 当特征为 cn + fhog 的话，就为 11 + 31 = 42 维
        self._tmpl = self.getFeatures(image, 1)
        # _prob 是初始化时的高斯响应图，也就是在目标框的中心位置响应值最大
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        # alpha 是线性回归系数，有两个通道分成实部和虚部
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
        self.train(self._tmpl, 1.0)

    # 初始化 hanning 窗口，函数只在第一帧被执行
    # 目的是采样时为不同的样本分配不同的权重，0.5 * 0.5 是用汉宁窗归一化为 [0, 1]，得到的矩阵值就是每个样本的权重
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hog_feature:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        # 相当于把 1d 的汉宁窗赋值成多个通道
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    # 标签制作，函数只在第一帧的时候执行（高斯响应）
    # 对于 ground_truth，由于模板函数的中心就是目标框的中心，因此论文中使用高斯分布函数作为标签，其分布函数为:
    # g(x, y) = exp((-1 / (2 * sigma * sigma)) * ((i - cx) ^ 2 + (j - cy)^2))
    # sigma = sqrt(sizeX * sizeY) / (padding * output_sigma_factor)
    # 其中，(cx, cy) 表示图像特征矩阵中心，padding 表示扩展框相对于目标框的变化比例为 2.5，output_sigma_factor 表示设定的一个值为 0.125
    def createGaussianPeak(self, sizey, sizex):
        # syh, sxh 为图像特征矩阵的中心点坐标
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        # 返回的标签 res 会进行快速傅里叶变换，也就是核相关滤波训练中的 y_hat
        return fftd(res)

    # 核相关矩阵计算提速，这里使用的是高斯核函数，其中 x1, x2 必须都是 M * N 的大小
    # 由于无论是训练还是检测都使用到了核相关矩阵，所以 hog 特征的融合主要是在这个过程中进行的，公式如下
    # k = exp(-d / (sigma * sigma))
    # d = max(0, (x1x1 + x2x2 - 2 * x1 * x2) / numel(x1))
    # numel(x1) 也就是 x1 的总像素点的个数
    def gaussianCorrelation(self, x1, x2):
        """
        我们首先把特征记为 x1[31]，x2[31]，是一个 31 维的容器，每个维度都是一个长度为 mn 的向量，也就是 x1 和 x2 的 shape 为 (31, mn)
        注意，这里的 m, n 也就是 size_patch[0] 和 size_patch[1]，所以 size_patch 的 shape 为 [m, n, 31]
        接下来第一步就是计算上面公式中的 x1 * x2
        1.首先分别对每个维度进行傅里叶变换，得到 xf1[31] 和 xf2[31]
        2.xf1[i] 以及 xf2[i] 表示是一个长度为 mn 的向量，所以将 xf1[i] 和 xf2[i] 转变为 [m,n] 的矩阵
        3.计算 xf1[i] 和 xf2[i] 的共轭在频域的点积（element-wise），这样得到的是 36 个 [m,n] 的复数矩阵，分别对每个矩阵都进行傅里叶逆变换得到 xf12[36],
        是 36 个 [m,n] 的实数矩阵，然后把 36 个矩阵对应点求和得到一个矩阵记作 xf12，是一个 [m,n] 的实数矩阵
        """
        if self._hog_feature or self._cn_feature:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                # 将 x1[i], x2[i] 转变为 [m,n] 的矩阵
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                # 傅立叶域点乘
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                # 进行傅立叶逆变换
                caux = real(fftd(caux, True))
                c += caux
            c = rearrange(c)
        else:
            # 'conjB=' is necessary!
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            # python 中，* 表示矩阵中对应元素的点乘，而不是矩阵乘法
            # np.sum(x1 * x1) 相当于求矩阵 x1 的二范数，然后将矩阵中的每一个元素累加起来，得到的就是一个实数，对于 x2 也是同理
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        # 等价于 d = max(d, 0)
        d = d * (d >= 0)
        # 得到核相关矩阵
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        # self._roi 表示初始的目标框 [x, y, width, height]
        extracted_roi = [0, 0, 0, 0]
        # cx, cy 表示目标框中心点的 x 坐标和 y 坐标
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float

        if inithann:
            # 保持初始目标框中心不变，将目标框的宽和高同时扩大相同倍数
            # 将目标框扩大 padding 倍是因为需要对目标框中的目标进行循环移位（x, y 两个方向）
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if self.template_size > 1:
                # 设定模板图像尺寸为 96，计算扩展框与模板图像尺寸的比例
                # 把最大的边缩小到 96，_scale 是缩小比例，_tmpl_sz 是滤波模板裁剪下来的 PATCH 大小
                # scale = max(w,h) / template
                self._scale = max(padded_h, padded_w) / float(self.template_size)
                # 同时将 scale 应用于宽和高，获取图像提取区域
                # roi_w_h = (w / scale, h / scale)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if self._hog_feature or self._cn_feature:
                # 由于后面提取 hog 特征时会以 cell 单元的形式提取，另外由于需要将频域直流分量移动到图像中心，因此需保证图像大小为 cell大小的偶数倍，
                # 另外，在 hog 特征的降维的过程中是忽略边界 cell 的，所以还要再加上两倍的 cell 大小
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        # 选取从原图中扣下的图片位置大小
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        # z 是当前被裁剪下来的搜索区域
        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz))

        # 如果同时使用了 fhog + raw_pixel 颜色特征
        if self._hog_feature and self._cn_feature:
            h, w = z.shape[:2]
            img = cv2.resize(z, (w + 2 * self.cell_size, h + 2 * self.cell_size))
            mapp = {'sizeX': 0, 'sizeY': 0, 'hogFeatures': 0, 'cnFeatures': 0, 'map': 0}
            # 对目标图像进行处理，获取到方向梯度直方图，mapp['map'] 的 shape 为 [sizeY, sizeX, 27]
            mapp = fhog.getFeatureMaps(img, self.cell_size, mapp)
            # 对目标图像的 cell 进行邻域归一化以及截断操作，得到的特征矩阵的 shape 为 [sizeY, sizeX, 108]，每一个 cell 的维度为 108 = 4 * 27 维
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            # 对目标图像进行 PCA 降维，将每一个 cell 的维度由 108 维变为 27 + 4 = 31 维，得到的特征矩阵的 shape 为 [sizeY, sizeX, 31]
            mapp = fhog.PCAFeatureMaps(mapp)

            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['hogFeatures']]))
            hog_feature = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T   # (size_patch[2], size_patch[0]*size_patch[1])

            cn_feature = extract_cn_feature(z, mapp, self.cell_size)
            mapp['cnFeatures'] = cn_feature.shape[0]
            FeaturesMap = np.concatenate((hog_feature, cn_feature), axis=0)

            # size_patch 为列表，保存裁剪下来的特征图的 [长，宽，通道]
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['hogFeatures'] + mapp['cnFeatures']]))

        # 如果只使用了 fhog 特征
        elif self._hog_feature:
            h, w = z.shape[:2]
            img = cv2.resize(z, (w + 2 * self.cell_size, h + 2 * self.cell_size))
            mapp = {'sizeX': 0, 'sizeY': 0, 'hogFeatures': 0, 'map': 0}
            # 对目标图像进行处理，获取到方向梯度直方图，mapp['map'] 的 shape 为 [sizeY, sizeX, 27]
            mapp = fhog.getFeatureMaps(img, self.cell_size, mapp)
            # 对目标图像的 cell 进行邻域归一化以及截断操作，得到的特征矩阵的 shape 为 [sizeY, sizeX, 108]，每一个 cell 的维度为 108 = 4 * 27 维
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            # 对目标图像进行 PCA 降维，将每一个 cell 的维度由 108 维变为 27 + 4 = 31 维，得到的特征矩阵的 shape 为 [sizeY, sizeX, 31]
            mapp = fhog.PCAFeatureMaps(mapp)

            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['hogFeatures']]))
            hog_feature = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T  # (size_patch[2], size_patch[0]*size_patch[1])
            FeaturesMap = hog_feature

        # 如果只使用了 raw_pixel 颜色特征
        elif self._cn_feature:
            h, w = z.shape[:2]
            img = cv2.resize(z, (w + 2 * self.cell_size, h + 2 * self.cell_size))
            mapp = {'sizeX': 0, 'sizeY': 0, 'cnFeatures': 0, 'map': 0}

            cn_feature = extract_cn_feature(img, mapp, self.cell_size)
            mapp['cnFeatures'] = cn_feature.shape[0]
            FeaturesMap = cn_feature
            inithann = False
            # size_patch 为列表，保存裁剪下来的特征图的 [长，宽，通道]
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['cnFeatures']]))

        # 如果既没有使用 raw_pixel 颜色特征也没有使用 fhog 特征，那么使用灰度图像特征，将 RGB 图像转变为单通道灰度图像
        else:
            if z.ndim == 3 and z.shape[2] == 3:
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)   # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
            elif z.ndim == 2:
                FeaturesMap = z  # (size_patch[0], size_patch[1]) # np.int8  #0~255
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            # size_patch 为列表，保存裁剪下来的特征图的 [长，宽，1]
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if inithann:
            self.createHanningMats()  # create Hanning Mats need size_patch
            # 加汉宁窗减少频谱泄漏
            FeaturesMap = self.hann * FeaturesMap

        return FeaturesMap

    # 根据上一帧结果使用快速检测公式计算当前帧中目标的位置，以及偏离采样中心的位移和峰值
    # z 是前一帧的训练（第一帧的初始化）模板 self._tmpl，x 是当前帧 fhog 特征，peak_value 是检测结果峰值
    def detect(self, z, x):
        # 使用高斯核函数求解核相关矩阵
        k = self.gaussianCorrelation(x, z)
        # 得到响应图，这里实际上就是快速检测公式，利用上一帧训练求得的线性回归参数 alpha 来进行检测
        # y_hat = k_hat * alpha，其中 * 表示对位相乘，并且利用转置消除共轭
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))

        # pv:响应最大值，pi:响应最大点的索引数组: (列下标，行下标)
        _, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int
        # 得到响应最大的点索引的 float 表示
        p = [float(pi[0]), float(pi[1])]   # [x,y], [float,float]

        # 使用幅值作差来定位峰值的位置
        # 也就是对于该响应矩阵，找出其最大响应值 peak_value 和最大响应位置 pxy，如果最大响应位置不在图像边界，那么
        # 分别比较最大响应位置两侧的响应大小，如果右侧比左侧高，或者下侧比上侧高，则分别将最大响应位置向较大的一侧移动一段距离
        # px = px + 0.5 * ((right - left) / (2 * peak_value - right - left))
        # py = py + 0.5 * ((down - up) / (2 * peak_value - down - up))
        if 0 < pi[0] < res.shape[1] - 1:
            p[0] += subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if 0 < pi[1] < res.shape[0] - 1:
            p[1] += subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        # 得出偏离采样中心的位移，res.shape[1] / 2 表示采样中心的 x 坐标，res.shape[0] / 2 表示采样中心的 y 坐标
        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        # 返回偏离采样中心的位移和峰值
        return p, pv

    # 使用当前图像的检测结果进行训练
    # x 是当前帧当前尺度下的 fhog 特征矩阵，train_interp_factor 是 interp_factor
    def train(self, x, train_interp_factor):
        # 使用高斯核函数计算核相关矩阵 k
        k = self.gaussianCorrelation(x, x)
        # 求解线性回归系数，alpha_hat = (1 / (k_hat + lamda)) * y_hat，求解出来的线性回归系数会在快速检测中使用到
        # 其中 y_hat 就是制作的响应标签（self._prob），由于 self._prob 在 createGaussianPeak 方法中返回时已经经过了 fft 变换，
        # 所以可以直接使用在公式中
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)

        # 模板更新: template = (1 - 0.012) * template + 0.012 * z
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        # 线性回归系数 self._alpha 的更新
        # alpha = (1 - 0.012) * alpha + 0.012 * alpha_x_z
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    # 获取当前帧的目标位置以及尺度，image 为当前帧的整幅图像
    # 基于当前帧更新目标位置
    def update(self, image):
        # roi 为 [x, y, width, height]
        roi = self._roi
        # 修正边界
        if roi[0] + roi[2] <= 0:
            roi[0] = -roi[2] + 1
        if roi[1] + roi[3] <= 0:
            roi[1] = -roi[2] + 1
        if roi[0] >= image.shape[1] - 1:
            roi[0] = image.shape[1] - 2
        if roi[1] >= image.shape[0] - 1:
            roi[1] = image.shape[0] - 2

        # 跟踪框、尺度框的中心
        cx = roi[0] + roi[2] / 2.
        cy = roi[1] + roi[3] / 2.
        # loc: 表示新的最大响应值偏离 roi 中心的位移
        # peak_value: 尺度不变时检测峰值结果
        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        # 略大尺度和略小尺度进行检测
        # 对于不同的尺度，都有着尺度惩罚系数 scale_weight，计算的公式如下：
        # _scale = _scale * (1 / scale_step)
        # T_w_h = T_w_h * (1 / scale_step)
        # T_x_y = T_cx_cy - T_w_h / 2 + res_x_y * cell_size * _scale

        if self.scale_step != 1:
            # Test at a smaller_scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
            # Test at a bigger_scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            # 在计算其他尺度的响应时，会乘以一个惩罚系数，并且会把 T_w_h 乘以 scale_step
            # 或者除以 scale_step 进行尺度缩小和扩大。self._scale 表示的是扩展框（padding 之后的图像）与模板图像尺寸的比例
            if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                roi[2] /= self.scale_step
                roi[3] /= self.scale_step
            elif self.scale_weight * new_peak_value2 > peak_value:
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                roi[2] *= self.scale_step
                roi[3] *= self.scale_step

        # 重新计算 roi[0] 和 roi[1] 使得新的最大响应值位于目标框的中心
        roi[0] = cx - roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        roi[1] = cy - roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if roi[0] >= image.shape[1] - 1:
            roi[0] = image.shape[1] - 1
        if roi[1] >= image.shape[0] - 1:
            roi[1] = image.shape[0] - 1
        if roi[0] + roi[2] <= 0:
            roi[0] = -roi[2] + 2
        if roi[1] + roi[3] <= 0:
            roi[1] = -roi[3] + 2

        assert (roi[2] > 0 and roi[3] > 0)

        return roi, peak_value

    def retrain(self, image, roi):
        self._roi = roi
        # 使用当前的检测框来训练样本参数
        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

def extract_image(image, roi):
    img = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return img