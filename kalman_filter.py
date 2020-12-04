# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        self.ndim = 4
        self.dt = 1.
        ndim = 4

        # Create Kalman filter model matrices.
        '''
        初始化状态转移矩阵 A 为:
        [[1, 0, 0, 0, dt, 0, 0, 0],
         [0, 1, 0, 0, 0, dt, 0, 0],
         [0, 0, 1, 0, 0, 0, dt, 0],
         [0, 0, 0, 0, 1, 0, 0, dt],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]]
         矩阵 A 中的 dt 是当前帧与前一帧之间的差（程序中取值为 1)，从这个矩阵可以看出 DEEP-SORT 使用的是一个匀速模型
        '''
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.dt

        '''
        初始化测量矩阵 H 为：
        [[1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0.]]
         H 将 track 的均值向量 [cx,cy,r,h,vx,vy,vr,vh] 映射到检测空间 [cx,cy,cr,ch]
        '''
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    # 对于未匹配到的目标新建一个 track，一般这样的检测目标都是新出现的物体
    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        # 输入 measurement 为一个检测目标的 box 🎒信息（center_x, center_y, w/h, height)，比如 [1348, 535, 0.3, 163]
        mean_pos = measurement
        # 刚出现的新目标默认其速度为 0，构造一个与 box 维度一样的向量 [0,0,0,0]
        mean_vel = np.zeros_like(mean_pos)
        # 按列连接两个矩阵 [1348, 535, 0.3, 163, 0, 0, 0, 0]，也就是初始化均值矩阵为 [center_x,center_y,ratio,h,vx,vy,vr,vh]
        mean = np.r_[mean_pos, mean_vel]

        # 协方差矩阵，元素值越大，表明不确定性越大，可以选择任意值初始化
        std = [2 * self._std_weight_position * measurement[3],  # 2 * 1/20 * h = 0.1 * h，高度缩小了 10 倍
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],  # 10 * 1/160 * h = h/16
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        # 主要根据目标的高度构造协方差矩阵
        # 对 std 中的每个元素平方，np.diag 构成一个 8*8 的对象矩阵，对角线上的元素是 np.square(std)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """

        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]

        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]

        # 初始化噪声矩阵 Q，代表了我们建立模型的不确定度，一般初始化为很小的值，这里是根据 track 的高度 h 初始化的 motion_cov
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        # x_prior(:k) = F * x_post(:k-1)
        mean = np.dot(self._motion_mat, mean)
        # p_prior(:k) = A * p_post(:k-1) * A.T + Q
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        # 返回的 mean 就代表了通过 kalman filter 预测得到的物体的下一个状态
        # 也就是说，在当前帧的检测与特征提取结束之后，因为并不知道原来每个 track 在当前帧的准确位置，先根据卡尔曼滤波去预测这些 track
        # 在当前帧的位置 mean 与 covariance，然后根据预测的均值和方差进行 track 和 detection 的匹配
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray，The state's mean vector (8 dimensional array).
        covariance : ndarray，The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)：Returns the projected mean and covariance matrix of the given state estimate.
        """

        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]

        # 这里计算的是检测器的噪声矩阵 R，它是一个 4*4 的对角矩阵，对角线上的值分别为中心点的两个坐标以及宽高的噪声，
        # 以任意值初始化，一般设置宽高的噪声大于中心点的噪声.
        innovation_cov = np.diag(np.square(std))
        # H * x_prior(:k)
        mean = np.dot(self._update_mat, mean)
        # H * p_prior(:k) * H.T
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        # mean = H * x_prior(:k)
        # H * p_prior(:k) * H.T + R
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        # 将 mean, covariance 映射到检测空间，作用就是把 1*8 的均值向量提取出了前面的 4 个位置向量 [cx,cy,r,h]
        projected_mean, projected_cov = self.project(mean, covariance)

        kalman_gain = np.linalg.multi_dot((covariance, np.transpose(self._update_mat), np.linalg.inv(projected_cov)))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        I = np.eye(2 * self.ndim)
        new_covariance = np.dot((I - np.dot(kalman_gain, self._update_mat)), covariance)

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """

        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
