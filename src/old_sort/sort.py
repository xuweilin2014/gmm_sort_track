from __future__ import print_function

from numba import jit
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import cv2


@jit
def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    # IoU = (bb_test 和 bb_gt 框相交部分的面积） / （bb_test 框面积 + bb_gt 框面积 - 两者相交部分的面积）
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
            bb_gt[3] - bb_gt[1]) - wh)
    return (o)


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


class Tracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, n_init=10):
        """
        Initialises a tracker using initial bounding box.
        使用初始化边界框初始化跟踪器
        """
        # define constant velocity model
        # 定义匀速模型，状态变量是 7 维，观测值是 4 维，按照需要的维度构建目标
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = Tracker.count
        Tracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.path = []
        self.path.append(bbox)

        self.n_init = n_init

        self.print_path = False

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.path.append(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def downward(self):
        if len(self.path) >= self.n_init:
            for i in range(len(self.path) - 1, len(self.path) - self.n_init + 1, -1):
                if self.path[i][3] - self.path[i - 1][3] < 10:
                    return False
            return True

        return False


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # 如果跟踪器为空
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    # 计算检测器与跟踪器 IoU 矩阵
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # 使用匈牙利算法进行 track 和 detection 之间的匹配
    # matched_indices 就是一个 (N,2) 矩阵，matched_indices[i] 表示一对匹配上的 (detection_index, track_index)
    matched_indices = linear_assignment(-iou_matrix)

    # 未匹配上的检测器 detection
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # 未匹配上的 track
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    # 过滤掉 IoU 比较小的 track 和 detection 之间的匹配对
    # matches 存放过滤后的匹配结果
    matches = []
    for m in matched_indices:
        # m[0] 是 detection index，m[1] 是 track index，如果它们之间的 IoU 小于阈值，则将它们视为未匹配
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # 将过滤后的维度变成 1*2 形式
        # 比如，m 是 [0,0]，那么通过 reshape 会变换成 [[0,0]]，[1,1] 会变成 [[1,1]]
        # 那么 matches 就会变成 [array([[0,0]]), array([[1,1]])]
        else:
            matches.append(m.reshape(1, 2))

    # 如果过滤后匹配将结果为空，那么返回空的匹配结果
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    # 如果过滤后的匹配结果非空，那么按 0 轴方向继续添加匹配对
    # numpy 提供了 numpy.concatenate((a1,a2,...), axis=0) 函数。能够一次完成多个数组的拼接。其中 a1,a2,... 是数组类型的参数
    # 比如，a = np.array([[1,2,3], [4,5,6]]), b = np.array([[11, 21, 31], [7,8,9]])
    # np.concatenate((a,b), axis=0) 结果为 array([[1,2,3], [4,5,6], [11, 21, 31], [7, 8, 9]])
    # 那么前面的 matches 就会变成 array([[0,0], [1,1]])
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


"""
SORT 跟踪算法到底在干什么？
1.假设 T1 时刻成功跟踪了某个单个物体，ID 为 1，绘制物体跟踪 BBox（紫色）
2.T2 时刻物体检测 BBox 总共有 4 个（黑色），预测T2时刻物体跟踪的 BBox（紫色）有1个，解决紫色物体跟踪BBox如何与黑色物体检测 BBox 关联的算法，就是 SORT 物体跟踪算法要解决的核心问题
3.SORT 关联两个 BBox 的核心算法是：用 IoU 计算 Bbox 之间的距离 + 匈牙利算法选择最优关联结果
"""


class Sort(object):
    def __init__(self, max_age=30, min_hits=10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    """
    :param dets: 输入的 dets 是检测结果，形式为 [x1, y1, x2, y2, score]
    """
    def update(self, dets):
        """
        Params:
        dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        # 帧计数
        self.frame_count += 1
        # get predicted locations from existing trackers.
        # 根据当前所有的卡尔曼跟踪器 tracker 的个数创建二维零矩阵，维度为：bbox + 卡尔曼跟踪器的 id
        trks = np.zeros((len(self.trackers), 5))
        # 存放待删除
        to_del = []
        # 存放最后返回的结果
        ret = []

        # 循环遍历 tracker 跟踪器的列表
        for t, trk in enumerate(trks):
            # 预测跟踪器 tracker 对应的物体在当前帧中的 bbox 位置
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果预测的 bbox 为空，那么将第 t 个 tracker 删除掉
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # 将预测为空的卡尔曼跟踪器所在行删除，最后 trks 中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空 bbox
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # 对 del 数组进行倒序遍历
        for t in reversed(to_del):
            # 从 tracker 列表中删除掉在这一帧图像中预测的 bbox 为空的 tracker
            self.trackers.pop(t)

        if dets.any():
            # 对传入的检测结果 dets 与上一帧跟踪物体在当前帧中预测的结果 trks 做关联，
            # 返回匹配的目标矩阵 matched, 新增目标的矩阵 unmatched_dets, 离开画面的目标矩阵 unmatched_trks
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            # 对卡尔曼跟踪器做遍历
            for t, trk in enumerate(self.trackers):
                # 如果上一帧中的 tracker 还在当前帧画面中（即不在当前预测的离开画面的矩阵 unmatched_trks 中）
                # 说明 tracker 在当前帧中找到了和其相匹配的 detection，接着在 matched 矩阵中找到与其关联的
                # 检测器的 bbox 结果，并用其来更新 tracker 中的卡尔曼滤波器
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])

            # create and initialise new trackers for unmatched detections
            # 对于新增的未匹配的检测结果，创建并初始化跟踪器
            for i in unmatched_dets:
                # 将新增的未匹配的检测结果 dets[i, :] 传入到 Tracker，从而重新创建一个 tracker
                trk = Tracker(dets[i, :])
                # 将刚刚创建和初始化的跟踪器 trk 传入到 trackers 中
                self.trackers.append(trk)

        i = len(self.trackers)
        # tracker_copy 用来保存 tracker，在图片上画出轨迹
        tracker_copy = []
        # 对这一帧中的 tracker 进行倒序遍历
        for trk in reversed(self.trackers):
            # 获取 tracker 跟踪器的状态 [x1, y1, x2, y2]
            d = trk.get_state()[0]
            # 同时满足以下两个条件的 tracker 才能够返回:
            # 1.tracker 没有匹配上 detection 的次数少于 max_age 次
            # 2.tracker 最近 10 帧连续下落，并且每一帧下落的距离满足要求
            if (trk.time_since_update < self.max_age) and trk.downward():
                # tracker 到目前为止至少连续匹配上了 min_hits 次 (除非当前为止的帧数少于 min_hits)，并且 tracker 只要有一次没有匹配上，
                # tracker 的 hit_streak 就会被重置为 0
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                tracker_copy.append(trk)
            i -= 1
            # remove dead tracker
            # 如果当前 tracker 已经有 max_age 次没有目标匹配上，则说明此 track 跟踪的物体已经离开了画面，所以将其从 tracker 集合中删除
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret), tracker_copy, trks

        return np.empty((0, 5)), tracker_copy, trks
