import numpy as np
from kcf_tracker import KCFTracker
from scale import ScaleEstimator
from scale_kalman_filter import ScaleKalmanFilter
from translation_kalman_filter import TranslationKalmanFilter
from utils import *


# 将 bbox 由 [x1,y1,x2,y2] 形式转为 [框中心点 x, 框中心点 y, 框面积 s, 宽高比例 r].T
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)

    # 将数组 [x,y,s,r] 转为 4 行 1 列形式，即 [x,y,s,r].T
    return np.array([x, y, s, r]).reshape((4, 1))


# 将 bbox 由 [x,y,s,r] 形式转为 [x1,y1,x2,y2] 形式
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, frame, frame_count, det, track_id, n_init=30, max_age=30, cn=True, hog=True, peak_threshold=0.4):
        """
        Initialises a tracker using initial bounding box.
        使用初始化边界框初始化跟踪器
        """
        bbox = det.to_tlbr()

        roi = list(map(float, det.to_tlwh()))
        self.scale = ScaleEstimator(roi, frame, hog)
        self.kcf = KCFTracker(roi, self.scale, cn, hog, peak_threshold)
        self.kcf.init(frame)

        self.track_id = track_id
        self.state = TrackState.Tentative

        self.hits = 1
        self.time_since_update = 0

        self.path = []
        self.path.append([frame_count, bbox[0], bbox[1], bbox[2], bbox[3]])
        self.n_init = n_init
        self.max_age = max_age

        self.tran_kf = TranslationKalmanFilter()
        self.tran_kf.initiate(det.to_xyah())

        self.scale_kf = ScaleKalmanFilter()
        self.scale_kf.initiate(det.to_xyah())

        self.expand_num = 1.05

        self.t = 1
        self.x_avg = np.zeros(2)
        # 在计算 mot 指标的时候使用
        self.print_path = False

    def update(self, frame, frame_count, det, roi):
        self.scale.set_roi(roi)

        scale_pi = self.scale.detect_scale(frame)
        factor = self.scale.scale_factors[scale_pi[0]] * self.scale.current_scale_factor
        cf_scale = np.array([self.scale.base_width, self.scale.base_height]) * factor
        det_scale = np.array(roi[2:])

        # 对【一维尺度滤波器】和【检测器】得到的大小，进行融合得到一个更加准确地物体尺度
        sc = self.weighted_fusion(2, cf_scale, det_scale)
        sc = sc[:] * self.expand_num
        self.scale.current_scale_factor = max(sc[0] / self.scale.base_width, sc[1] / self.scale.base_height)
        if self.scale.current_scale_factor < self.scale.min_scale_factor:
            self.scale.current_scale_factor = self.scale.min_scale_factor

        self.scale.train_scale(frame)
        _, _, width, height = self.scale.get_roi()
        measurements = np.r_[roi[:2], width / height, height]

        # 更新卡尔曼滤波的状态方程
        self.tran_kf.update(measurements)
        self.scale_kf.update(measurements)

        # 使用当前的检测框来训练滤波器的参数
        x = self.kcf.getFeatures(frame, 0, 1.0, self.scale.get_roi())
        self.kcf.train(x, self.kcf.interp_factor)

        self.hits += 1
        self.time_since_update = 0

        tlbr = det.to_tlbr()
        self.path.append([frame_count, tlbr[0], tlbr[1], tlbr[2], tlbr[3]])

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.tran_kf.predict()
        self.scale_kf.predict()
        self.time_since_update += 1

    # 返回跟踪器的状态 [top_x, top_y, bottom_x, bottom_y]
    def to_tlbr(self):
        ret = np.r_[self.tran_kf.mean[:2], self.scale_kf.mean[:2]].copy()
        ret[2] *= ret[3]
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_tlwh(self):
        ret = self.to_tlbr().copy()
        ret[2:] -= ret[:2]
        return ret

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def downward(self):
        if len(self.path) >= self.n_init:
            for i in range(len(self.path) - 1, len(self.path) - self.n_init + 1, -1):
                if self.path[i][4] - self.path[i - 1][4] < 10:
                    return False
            return True
        return False

    def weighted_fusion(self, sensor_num, cf_scale, det_scale):
        res = []

        for vals in zip(cf_scale, det_scale):
            r_mat = np.zeros((sensor_num, sensor_num))
            for i in range(sensor_num):
                for j in range(sensor_num):
                    r_mat[i][j] = ((self.t - 1.0) / self.t) * r_mat[i][j] + (1.0 / self.t) * vals[i] * vals[j]

            new_sigma = np.zeros(sensor_num)
            sigma_sum = 0
            for i in range(sensor_num):
                new_sigma[i] = r_mat[i][i] - (1.0 / (sensor_num - 1)) * (np.sum(r_mat[i, :]) - r_mat[i][i])
                new_sigma[i] = 1.19209e-07 if new_sigma[i] <= 0 else new_sigma[i]
                self.x_avg[i] = ((self.t - 1.0) / self.t) * self.x_avg[i] + (1.0 / self.t) * vals[i]
                sigma_sum += 1.0 / new_sigma[i]

            weight = 1 / (new_sigma * sigma_sum)
            res.append(int(np.sum(weight * self.x_avg)))

        self.t += 1
        return np.array(res, dtype=int)