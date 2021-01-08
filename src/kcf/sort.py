from __future__ import print_function
import numpy as np
from associate import associate_detections_to_trackers, kcf_associate
from track import Track
from src.gmm_sort import kalman_filter

"""
SORT 跟踪算法到底在干什么？
1.假设 T1 时刻成功跟踪了某个单个物体，ID 为 1，绘制物体跟踪 BBox（紫色）
2.T2 时刻物体检测 BBox 总共有 4 个（黑色），预测T2时刻物体跟踪的 BBox（紫色）有1个，解决紫色物体跟踪BBox如何与黑色物体检测 BBox 关联的算法，就是 SORT 物体跟踪算法要解决的核心问题
3.SORT 关联两个 BBox 的核心算法是：用 IoU 计算 Bbox 之间的距离 + 匈牙利算法选择最优关联结果
"""


class Sort(object):
    def __init__(self, max_age=30, min_hits=10, n_init = 10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0
        self.kf = kalman_filter.KalmanFilter()
        self.global_id = 1
        self.n_init = 10

    # 将 (top_x, top_y, bottom_x, bottom_x, bottom_y) 转变为 (center_x, center_y, aspect ratio, h)
    # aspect ratio = width / height
    @staticmethod
    def to_xyah(vector):
        w = vector[2] - vector[0]
        h = vector[3] - vector[1]
        ret = vector.copy()
        ret[:2] = (ret[:2] + ret[2:]) / 2
        ret[2] = w / h
        ret[3] = h

        return ret

    def generate_id(self):
        copy = self.global_id
        self.global_id += 1
        return copy

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    """
    :param dets: 输入的 dets 是检测结果，形式为 [x1, y1, x2, y2, score]
    """
    def update(self, frame, detections):
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
        trks = np.zeros((len(self.tracks), 5))
        dets = []
        # 存放待删除
        to_del = []
        # 存放最后返回的结果
        ret = []

        for det in detections:
            dets.append(det.to_tlbr())
        dets = np.asarray(dets)

        # 循环遍历 tracker 跟踪器的列表
        for t, trk in enumerate(trks):
            # 预测跟踪器 tracker 对应的物体在当前帧中的 bbox 位置
            pos = self.tracks[t].to_tlbr()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果预测的 bbox 为空，那么将第 t 个 track 删除掉
            if np.any(np.isnan(pos)) :
                to_del.append(t)

        # 将预测为空的卡尔曼跟踪器所在行删除，最后 trks 中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空 bbox
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # 对 del 数组进行倒序遍历
        for t in reversed(to_del):
            # 从 tracker 列表中删除掉在这一帧图像中预测的 bbox 为空的 track
            self.tracks.pop(t)

        if dets.any():
            # 对传入的检测结果 dets 与上一帧跟踪物体在当前帧中预测的结果 trks 做关联，
            # 返回匹配的目标矩阵 matched, 新增目标的矩阵 unmatched_dets, 离开画面的目标矩阵 unmatched_trks
            matched_a, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
            matched_b, unmatched_dets, unmatched_trks = kcf_associate(frame, unmatched_dets, unmatched_trks, detections, self.tracks)
            matched = np.concatenate((matched_a, matched_b), axis=0)

            # update matched trackers with assigned detections
            # 对卡尔曼跟踪器做遍历
            for detection_idx, track_idx in matched:
                # 如果上一帧中的 tracker 还在当前帧画面中（即不在当前预测的离开画面的矩阵 unmatched_trks 中）
                # 说明 tracker 在当前帧中找到了和其相匹配的 detection，接着在 matched 矩阵中找到与其关联的
                # 检测器的 bbox 结果，并用其来更新 tracker 中的卡尔曼滤波器
                trk = self.tracks[track_idx]
                detection = detections[detection_idx]
                trk.update(frame, self.kf, detection.to_tlwh(), detection.to_tlbr(), detections[detection_idx].to_xyah())

            for track_idx in unmatched_trks:
                self.tracks[track_idx].mark_missed()

            # create and initialise new trackers for unmatched detections
            # 对于新增的未匹配的检测结果，创建并初始化跟踪器
            for detection_idx in unmatched_dets:
                det = detections[detection_idx]
                mean, covariance = self.kf.initiate(det.to_xyah())
                # 将新增的未匹配的检测结果 dets[i, :] 传入到 Tracker，从而重新创建一个 tracker
                trk = Track(det.to_tlbr(), mean, covariance, self.generate_id(), self.n_init, self.max_age)
                # 初始化 tracker 中的 kcf 跟踪器
                trk.init_kcf(frame, det.tlwh)
                # 将刚刚创建和初始化的跟踪器 trk 传入到 trackers 中
                self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        i = len(self.tracks)
        # tracker_copy 用来保存 tracker，在图片上画出轨迹
        tracker_copy = []
        # 对这一帧中的 tracker 进行倒序遍历
        for trk in reversed(self.tracks):
            # 获取 tracker 跟踪器的状态 [x1, y1, x2, y2]
            d = trk.to_tlbr()
            # 同时满足以下两个条件的 tracker 才能够返回:
            # 1.tracker 没有匹配上 detection 的次数少于 max_age 次
            # 2.tracker 最近 10 帧连续下落，并且每一帧下落的距离满足要求
            if (trk.time_since_update < self.max_age) and trk.downward():
                # tracker 到目前为止至少连续匹配上了 min_hits 次 (除非当前为止的帧数少于 min_hits)，并且 tracker 只要有一次没有匹配上，
                # tracker 的 hit_streak 就会被重置为 0
                if (trk.hits >= self.min_hits or self.frame_count <= self.min_hits) and trk.time_since_update == 0:
                    ret.append(np.concatenate((d, [trk.track_id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                tracker_copy.append(trk)
            i -= 1
            # remove dead tracker
            # 如果当前 tracker 已经有 max_age 次没有目标匹配上，则说明此 track 跟踪的物体已经离开了画面，所以将其从 tracker 集合中删除
            # if trk.time_since_update > self.max_age:
            #     self.tracks.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret), tracker_copy, trks

        return np.empty((0, 5)), tracker_copy, trks
