from kalman_filter import *


# noinspection PyAttributeOutsideInit
class TranslationKalmanFilter(KalmanFilter):

    """
    位置卡尔曼滤波器，用来对物体的具体位置进行预测和更新
    位置卡尔曼滤波器使用一个 6 维的状态向量：[x, y, h, vx, vy, vh]，其中 h 就是检测框的 height 高度，这里加上一个 h 是为了后面
    初始化噪声矩阵方便
    对于 x, y 和 h 的预测，遵从一个匀加速模型
    """

    def __init__(self):
        super().__init__()
        self.ndim = 3
        self.dt = 1.
        ndim = 3

        # create kalman filter model matrices.
        '''
        初始化状态转移矩阵 A 为:
        [[1, 0, 0, dt, 0, 0],
         [0, 1, 0, 0, dt, 0],
         [0, 0, 1, 0, 0, dt],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
         矩阵 A 中的 dt 是当前帧与前一帧之间的差（程序中取值为 1)，从这个矩阵可以看出 DEEP-SORT 使用的是一个匀速模型
        '''
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.dt

        # 这个卡尔曼滤波器用来对物体的尺度大小进行预测
        self._control_mat = np.transpose(np.array([0.5, 0.5, 0.005, 1, 2.8, 0.000]))
        self.u = 1

        '''
        初始化测量矩阵 H 为：
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0]]
         H 将 track 的均值向量 [x, y, h, vx, vy, vh] 映射到检测空间 [x, y, h]
        '''
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    # 对于未匹配到的目标新建一个 track，一般这样的检测目标都是新出现的物体
    def initiate(self, measurement):
        # 输入 measurement 为一个检测目标的 box 信息（center_x, center_y, w/h, height)，比如 [1348, 535, 0.3, 163]
        # 这里我们取 measurement 向量中的第 0,1,3 个值来构成一个尺度向量 [x, y, height]
        mean_scale = np.r_[measurement[:2], measurement[3]]
        # 刚出现的新目标默认其速度为 0，构造一个与 box 维度一样的向量 [0, 0]
        mean_vel = np.zeros_like(mean_scale)
        # 按列连接两个矩阵 [1348, 535, 163, 0, 0, 0]，也就是初始化均值矩阵为 [x, y, h, vx, vy, yh]
        mean = np.r_[mean_scale, mean_vel]

        # 协方差矩阵，元素值越大，表明不确定性越大，可以选择任意值初始化
        std = [
            2 * self._std_weight_position * measurement[3],  # 2 * 1/20 * h = 0.1 * h，高度缩小了 10 倍
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],  # 10 * 1/160 * h = h/16
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3]
        ]

        # 主要根据目标的高度构造协方差矩阵
        # 对 std 中的每个元素平方，np.diag 构成一个 6*6 的对象矩阵，对角线上的元素是 np.square(std)
        covariance = np.diag(np.square(std))

        self.mean = mean
        self.covariance = covariance

        return mean, covariance

    def predict(self):
        # 输入的 mean 的值为 [x, y, h, vx, vy, vh]
        std = [
            self._std_weight_position * self.mean[2],
            self._std_weight_position * self.mean[2],
            self._std_weight_position * self.mean[2],
            self._std_weight_velocity * self.mean[2],
            self._std_weight_velocity * self.mean[2],
            self._std_weight_velocity * self.mean[2]
        ]

        # 初始化噪声矩阵 Q，代表了我们建立模型的不确定度，一般初始化为很小的值，这里是根据 track 的高度 h 初始化的 motion_cov
        # motion_cov 最后被初始化为一个 6*6 的对角矩阵
        motion_cov = np.diag(np.square(std))
        # x_prior(:k) = F * x_post(:k-1)
        self.mean = np.dot(self._motion_mat, self.mean) + self._control_mat * self.u
        # p_prior(:k) = A * p_post(:k-1) * A.T + Q
        self.covariance = np.linalg.multi_dot((self._motion_mat, self.covariance, self._motion_mat.T)) + motion_cov

        # 返回的 mean 就代表了通过 kalman filter 预测得到的物体的下一个状态
        # 也就是说，在当前帧的检测与特征提取结束之后，因为并不知道原来每个 track 在当前帧的准确位置，先根据卡尔曼滤波去预测这些 track
        # 在当前帧的位置 mean 与 covariance，然后根据预测的均值和方差进行 track 和 detection 的匹配

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[2],
               self._std_weight_position * mean[2],
               self._std_weight_position * mean[2]]

        # 这里计算的是检测器的噪声矩阵 R，它是一个 3*3 的对角矩阵，对角线上的值分别为中心点的两个坐标以及宽高的噪声，
        # 以任意值初始化，一般设置宽高的噪声大于中心点的噪声.
        innovation_cov = np.diag(np.square(std))
        # H * x_prior(:k)
        mean = np.dot(self._update_mat, mean)
        # H * p_prior(:k) * H.T
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        # mean = H * x_prior(:k)
        # H * p_prior(:k) * H.T + R
        return mean, covariance + innovation_cov

    def update(self, measurement):
        # measurement 表示的是一个 [x, y, r, h]
        measurement = np.r_[measurement[:2], measurement[3]]
        # 将 mean, covariance 映射到检测空间，作用就是把 1*6 的均值向量提取出了前面的 3 个位置向量 [x, y, h]
        projected_mean, projected_cov = self.project(self.mean, self.covariance)

        # covariance * _update_mat.T * (projected_cov)^(-1)
        # p_prior(:k) * H.T * (H * p_prior(:k) * H.T + R)^(-1)
        kalman_gain = np.linalg.multi_dot((self.covariance, np.transpose(self._update_mat), np.linalg.inv(projected_cov)))
        # x_post(:k) = x_prior(:k) + K(:k) * (z(:k) - H * x_prior(:k))
        new_mean = self.mean + np.dot(kalman_gain, measurement - projected_mean)

        I = np.eye(self.ndim * 2)
        # p_post(:k) = (I - K(:k) * H) * p_prior(:k)
        new_covariance = np.dot((I - np.dot(kalman_gain, self._update_mat)), self.covariance)

        self.mean = new_mean
        self.covariance = new_covariance

