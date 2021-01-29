import multiprocessing as mp

import cv2
import numpy as np

"""
下面的代码实现了自适应的高斯混合算法，也就是图像中每一个像素点的高斯模型个数都不相同，在更新的过程中，会对像素点的高斯模型个数进行动态删减。
在实现中加入了并行化，也就是一张图像的不同行的像素都由不同的进程来分别进行处理，加快运行速度
代码的实现参考以下论文：
adaptive Gaussian mixture model for real-time tracking
Improved Adaptive Gaussian Mixture Model for Background Subtraction
还有以下博客：
https://blog.csdn.net/lwx309025167/article/details/78538714
https://blog.csdn.net/lwx309025167/article/details/78554082
https://blog.csdn.net/abc20002929/article/details/43247425?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control
最后还参考了 opencv 中混合高斯建模的源代码，也就是 bgfg_gaussmix2.cpp
"""


# alpha 就是论文中的更新速率 learning rate，alpha = 1 / defaultHistory2
# defaultHistory2 表示训练得到背景模型所用到的集合大小，默认为 500，并且如果不手动设置 learning rate 的话，这个变量 defaultHistory2 就被用于计算当前的
# learning rate，此时 defaultHistory2 越大，learning rate 就越小，背景更新也就越慢
default_history = 500
# 方差阈值，用于判断当前像素是前景还是背景
default_var_threshold = 4.0 * 4.0
# 每个像素点高斯模型的最大个数，默认为 5
default_nmixtures = 5
# 高斯背景模型权重和阈值，nmixtures 个模型按权重排序之后，只取模型权重累加值大于 backgroundRatio 的前几个作为背景模型，也就是说如果该值取得非常小，很可能只使用
# 权重最大的高斯模型作为背景 (因为仅仅一个模型权重就大于 backgroundRatio)
default_background_ratio = 0.9
# 方差阈值，用于是否存在匹配的模型，如果不存在则新建一个
default_var_threshold_gen = 3.0 * 3.0
# 初始化一个高斯模型时，方差的值默认为 15
default_var_init = 15.0
# 高斯模型中方差的最大值为 5 * 15 = 75
default_var_max = 5 * default_var_init
# 高斯模型中方差的最小值为 4
default_var_min = 4.0
# 论文中提到的减常量，通过后验估计引入一个减常量，这是这个算法的最大创新点，之后根据权值为负时会把这个分布舍弃掉，实现分布数量的自适应性
# 如果将其设置为 0，那么算法就会变成标准的 Stauffer&Grimson 高斯混合背景建模算法
default_ct = 0.05

CV_CN_MAX = 512

FLT_EPSILON = 1.19209e-07


class GuassInvoker():
    def __init__(self, image, mask, gmm_model, mean_model, gauss_modes, nmixtures, lr, Tb, TB, Tg, var_init, var_min, var_max, prune, nchannels):
        self.image = image
        self.mask = mask
        self.gmm_model = gmm_model
        self.mean_model = mean_model
        self.gauss_modes = gauss_modes
        self.nmixtures = nmixtures
        self.lr = lr
        self.Tb = Tb
        self.TB = TB
        self.Tg = Tg
        self.var_init = var_init
        self.var_min = var_min
        self.var_max = var_max
        self.prune = prune
        self.nchannels = nchannels

    # 针对原图像中的某一行进行处理
    def calculate(self, row):
        lr = self.lr
        gmm_model = self.gmm_model
        mean_model = self.mean_model
        data = self.image[row]
        cols = data.shape[0]
        d_data = []

        # 遍历原图像中的某一行的所有列
        for col in range(cols):
            backgroud = False
            fits = False
            # 当前像素点的高斯模型个数
            modes_used = self.gauss_modes[row][col]
            total_weight = 0.

            # 当前像素点使用的所有高斯模型
            gmm_per_pixel = gmm_model[row][col]
            # 当前像素点使用的所有高斯模型的均值
            mean_per_pixel = mean_model[row][col]
            # 遍历每一个像素点的所有高斯模型
            for mode in range(modes_used):
                # 当前像素点的第 mode 个高斯模型，是一个长度为 2 的向量，[weight, variance]
                gmm = gmm_model[row][col][mode]
                # 当前像素点的第 mode 个高斯模型的均值，长度为 nchannels，也就是图像的分量
                mean = mean_model[row][col][mode]
                # 计算当前高斯分布的新权重，计算的公式如下：
                # weight = (1 - lr) * weight - lr * ct + lr * o
                # 其中 o 为 1 当且仅当当前像素点属于第 mode 个高斯分布，否则 o 为 0，因此 lr * o 必须在后面判断是否属于当前这个高斯分布之后，再加上 lr
                weight = (1 - lr) * gmm[0] + self.prune
                swap_count = 0

                if not fits:
                    var = gmm[1]
                    # 马氏距离
                    dist2 = 0
                    d_data = mean - data[col]
                    dist2 = np.sum(d_data ** 2)
                    # 使用马氏距离来判断当前像素点是属于前景还是背景
                    if total_weight < self.TB and dist2 < self.Tb * var:
                        backgroud = True
                    # 判断当前像素点是否属于当前这个高斯模型
                    if dist2 < self.Tg * var:
                        fits = True

                        # 当前像素点属于这个高斯模型，因此前面提到过的 o 值为 1，因此 weight 要加上 lr * o = lr
                        weight += lr
                        k = lr / weight
                        # 更新当前高斯模型的均值
                        mean -= k * d_data

                        # 更新当前高斯模型的方差
                        var += k * (dist2 - var)
                        var = max(var, self.var_min)
                        var = min(var, self.var_max)
                        gmm[1] = var

                        for i in range(mode, 0, -1):
                            if weight < gmm_per_pixel[i - 1][0]:
                                break
                            swap_count += 1
                            gmm_per_pixel[i - 1], gmm_per_pixel[i] = gmm_per_pixel[i], gmm_per_pixel[i - 1]
                            mean_per_pixel[i - 1], mean_per_pixel[i] = mean_per_pixel[i], mean_per_pixel[i - 1]

                # 保证下一次模型的权重非负，prune = lr * ct，也就是 weight 要大于常量值 prune，否则当前高斯模型的 weight 可能就会出现小于 0 的情况。接着将当前像素点的 nmodes 减一
                if weight < -self.prune:
                    weight = 0.
                    modes_used -= 1

                # 更新权重
                gmm_per_pixel[mode - swap_count][0] = weight
                total_weight += weight

            inv_factor = 0.
            if abs(total_weight) > FLT_EPSILON:
                inv_factor = 1.0 / total_weight

            gmm_per_pixel[:, 0] *= inv_factor

            if not fits and lr > 0:
                # 新增一个高斯模型或者替换掉权值最小的那个高斯模型
                if modes_used == self.nmixtures:
                    mode = self.nmixtures - 1
                else:
                    mode = modes_used
                    modes_used += 1

                # 如果只有一个高斯模型，那么就把这个高斯模型的权重设置为 1
                if modes_used == 1:
                    gmm_per_pixel[mode][0] = 1.0
                else:
                    # 新增加模型的权重等于 lr，也就是 learning rate
                    # 当前像素点有 nmixtures 个高斯模型，并且这些高斯模型是按照权重大小降序排列的
                    gmm_per_pixel[mode][0] = lr

                    # 归一化其他模型的权重
                    # 比如现在某一点的高斯混合模型有 3 个，权重分别为 0.1, 0.2, 0.7，而 learning rate 为 0.04，因此新模型的权重为 0.04，
                    # 那么接下来就会把 3 个高斯模型的值分别乘以 0.96，也就是 0.096，0.192，0.672，然后再加上 0.04 就等于 1
                    gmm_per_pixel[:, 0] *= (1 - lr)

                # 初始化新的高斯模型的均值 mean，使用的就是原始图像中的像素点的值来进行初始化
                mean_per_pixel[mode] = data[col]
                # 初始化新增的混合高斯模型 gmm 的方差 variance
                gmm_per_pixel[mode][1] = self.var_init

                # 对所有的高斯模型按照权重进行降序排序
                for i in range(modes_used - 1, 0, -1):
                    if lr < gmm_per_pixel[i - 1][0]:
                        break
                    gmm_per_pixel[i - 1], gmm_per_pixel[i] = gmm_per_pixel[i], gmm_per_pixel[i - 1]
                    mean_per_pixel[i - 1], mean_per_pixel[i] = mean_per_pixel[i], mean_per_pixel[i - 1]

            self.gauss_modes[row][col] = modes_used
            self.mask[row][col] = 0 if backgroud else 255

        return row, self.mask[row], self.gmm_model[row], self.mean_model[row], self.gauss_modes[row]


# noinspection PyAttributeOutsideInit
class GuassMixBackgroundSubtractor():
    def __init__(self):
        self.frame_count = 0
        self.history = default_history
        self.var_threshold = default_var_threshold
        self.nmixtures = default_nmixtures
        self.var_init = default_var_init
        self.var_max = default_var_max
        self.var_min = default_var_min
        self.var_threshold_gen = default_var_threshold_gen
        self.ct = default_ct
        self.background_ratio = default_background_ratio

    def apply(self, image, lr=-1):
        if self.frame_count == 0 or lr >= 1:
            self.initialize(image)

        self.image = image
        self.frame_count += 1
        # 计算 learning rate，也就是 lr，有以下三种情况：
        # 1.输入 lr 为 -1，那么 lr 就按照 history 的值来计算
        # 2.输入 lr 为 0，那么 lr 就按照 0 来计算，也就是说背景模型停止更新
        # 3.输入 lr 在 0 ~ 1 之间，那么背景模型更新速度为 lr，lr 越大更新越快，算法内部表现为当前帧参与背景更新的权重越大
        self.lr = lr if lr >= 0 and self.frame_count > 1 else 1 / min(2 * self.frame_count, self.history)
        print(self.lr)
        pool = mp.Pool(int(mp.cpu_count()))
        self.mask = np.zeros(image.shape[:2], dtype=int)
        # 对原图像中的每一行进行并行计算
        result = pool.map_async(self.parallel, [i for i in range(self.image.shape[0])]).get()
        pool.close()
        pool.join()
        # 计算完成之后再进行组合，得到最后的结果
        for row, mask_row, gmm_model_row, mean_model_row, gauss_modes_row in result:
            self.mask[row] = mask_row
            self.gauss_modes[row] = gauss_modes_row
            self.mean_model[row] = mean_model_row
            self.gmm_model[row] = gmm_model_row
        return self.mask

    def parallel(self, row):
        invoker = GuassInvoker(self.image, self.mask, self.gmm_model, self.mean_model, self.gauss_modes, self.nmixtures, self.lr,
                               self.var_threshold, self.background_ratio, self.var_threshold_gen, self.var_init,
                               self.var_min, self.var_max, float(-self.lr * self.ct), self.nchannels)
        return invoker.calculate(row)

    def initialize(self, image):
        # gauss_modes 这个矩阵用来存储每一个像素点使用的高斯模型的个数，初始的时候都为 0
        self.gauss_modes = np.zeros(image.shape[:2], dtype=np.int)
        height, width = image.shape[:2]
        if len(image.shape) == 2:
            self.nchannels = 1
        else:
            self.nchannels = image.shape[2]

        # 高斯混合背景模型分为两部分：
        # 第一部分：height * width * nmixtures (=5) * 2 * sizeof(float)，2 表示包含 weight 和 mean 两个 float 变量，也就是 gmm_model
        # 第二部分：height * width * nmixtures (=5) * nchannels * sizeof(float)。nchannels 一般为 3，表示 B, G, R 三个变量，其实也就是 mean 每个像素通道均对应一个均值，
        # 刚好有 nchannels 个单位的 float 大小，也就是 mean_model
        self.gmm_model = np.zeros((height, width, self.nmixtures, 2), dtype=np.float)
        self.mean_model = np.zeros((height, width, self.nmixtures, self.nchannels), dtype=np.float)


if __name__ == '__main__':
    img = cv2.imread('blank.png')
    cap = cv2.VideoCapture("/home/xwl/PycharmProjects/gmm-sort-track/input/mot.avi")

    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')

    mog = GuassMixBackgroundSubtractor()
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
            cv2.imwrite('./mask' + str(frame_count) + '.jpg', mask)
            print('writing mask' + str(frame_count) + '.jpg...')

