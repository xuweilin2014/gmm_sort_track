import numpy as np
import kalman_filter as kf


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

    def __init__(self, bbox, mean, covariance, track_id, n_init=10, max_age=30):
        """
        Initialises a tracker using initial bounding box.
        使用初始化边界框初始化跟踪器
        """
        # define constant velocity model
        # 定义匀速模型，状态变量是 8 维，观测值是 4 维，按照需要的维度构建目标
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.state = TrackState.Tentative

        self.hits = 1
        self.time_since_update = 0



        self.path = []
        self.path.append(bbox)
        self.n_init = n_init
        self.max_age = max_age

    def update(self, kf, tlbr, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, bbox)
        self.hits += 1
        self.time_since_update = 0
        self.path.append(tlbr)
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def predict(self, kf):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    # 返回跟踪器的状态 [top_x, top_y, bottom_x, bottom_y]
    def to_tlbr(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        ret[2:] = ret[:2] + ret[2:]
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
                if self.path[i][3] - self.path[i - 1][3] < 10:
                    return False
            return True

        return False
