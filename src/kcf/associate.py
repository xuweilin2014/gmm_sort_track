import numpy as np
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import *

INFTY_COST = -1e+5

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
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def comp_distance(det_box, kcf_box):
    cx = kcf_box[0] + kcf_box[2] / 2
    cy = kcf_box[1] + kcf_box[3] / 2
    det_cx = det_box[0] + det_box[2] / 2
    det_cy = det_box[1] + det_box[3] / 2
    return np.sqrt((cx - det_cx) ** 2 + (cy - det_cy) ** 2)

def kcf_associate(frame, match_d_t, unmatched_dets, unmatched_trks, dets, trks):
    unmatched_dets_final = []
    unmatched_trks_final = []
    peak_threshold = 0.4
    matched = []

    res_matrix = np.zeros((len(unmatched_dets), len(unmatched_trks)), dtype=np.float)
    roi_matrix = np.zeros((len(dets), len(trks)), dtype=object)

    for det, trk in match_d_t:
        roi = list(map(float, det.to_tlwh()))
        kcf = trk.kcf
        scale = trk.scale
        correct_bounding(roi, frame)
        # 跟踪框、尺度框的中心
        cx = roi[0] + roi[2] / 2.
        cy = roi[1] + roi[3] / 2.
        # loc:表示新的最大响应值偏离 roi 中心的位移
        # peak_value:尺度不变时检测峰值结果
        loc, peak_value = kcf.detect(kcf.tmpl, kcf.getFeatures(frame, 0, 1.0, roi))
        # 重新计算 roi[0] 和 roi[1] 使得新的最大响应值位于目标框的中心
        roi[0] = cx - roi[2] / 2.0 + loc[0] * kcf.cell_size * kcf.scale * scale.current_scale_factor
        roi[1] = cy - roi[3] / 2.0 + loc[1] * kcf.cell_size * kcf.scale * scale.current_scale_factor
        correct_bounding(roi, frame)

        d,t = np.argwhere(dets == det)[0][0], np.argwhere(trks == trk)[0][0]
        roi_matrix[d][t] = np.array(roi)

    for trk_index in range(len(unmatched_trks)):
        trk = unmatched_trks[trk_index]
        kcf = trk.kcf
        scale = trk.scale

        for det_index in range(len(unmatched_dets)):
            det = unmatched_dets[det_index]
            roi = list(map(float, det.to_tlwh()))

            # 修正边界
            correct_bounding(roi, frame)
            # 跟踪框、尺度框的中心
            cx = roi[0] + roi[2] / 2.
            cy = roi[1] + roi[3] / 2.
            # loc: 表示新的最大响应值偏离 roi 中心的位移
            # peak_value: 尺度不变时检测峰值结果
            loc, peak_value = kcf.detect(kcf.tmpl, kcf.getFeatures(frame, 0, 1.0, roi))
            # 重新计算 roi[0] 和 roi[1] 使得新的最大响应值位于目标框的中心
            roi[0] = cx - roi[2] / 2.0 + loc[0] * kcf.cell_size * kcf.scale * scale.current_scale_factor
            roi[1] = cy - roi[3] / 2.0 + loc[1] * kcf.cell_size * kcf.scale * scale.current_scale_factor
            correct_bounding(roi, frame)

            d,t = np.argwhere(dets == det)[0][0], np.argwhere(trks == trk)[0][0]
            res_matrix[det_index][trk_index] = peak_value
            roi_matrix[d][t] = np.array(roi)

    matched_indices = linear_assignment(-res_matrix)

    for t in unmatched_trks:
        if t not in unmatched_trks[matched_indices[:, 1]]:
            unmatched_trks_final.append(t)

    for d in unmatched_dets:
        if d not in unmatched_dets[matched_indices[:, 0]]:
            unmatched_dets_final.append(d)

    for m in matched_indices:
        flag = True
        det = unmatched_dets[m[0]]
        trk = unmatched_trks[m[1]]

        if res_matrix[m[0], m[1]] < peak_threshold:
            flag = False
        else:
            dist = comp_distance(det.to_tlbr(), trk.to_tlbr())
            if dist > np.min(trk.to_tlwh()[2:]) * 2:
                flag = False
            else:
                matched.append(np.r_[det, trk].reshape((1,2)))

        if not flag:
            unmatched_dets_final.append(det)
            unmatched_trks_final.append(trk)

    if len(matched) == 0:
        matched = np.empty((0, 2), dtype=object)
    else:
        matched = np.concatenate(matched, axis=0)

    return matched, unmatched_dets_final, unmatched_trks_final, roi_matrix


def associate_detections_to_trackers(detections, trackers, det_objs, trk_objs, iou_threshold=0.01):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # 如果跟踪器为空
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=object), det_objs, np.array([])
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
    for d, det in enumerate(det_objs):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(det)

    # 未匹配上的 track
    unmatched_trackers = []
    for t, trk in enumerate(trk_objs):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(trk)

    # filter out matched with low IoU
    # 过滤掉 IoU 比较小的 track 和 detection 之间的匹配对
    # matches 存放过滤后的匹配结果
    matches = []
    for m in matched_indices:
        det = det_objs[m[0]]
        trk = trk_objs[m[1]]
        # m[0] 是 detection index，m[1] 是 track index，如果它们之间的 IoU 小于阈值，则将它们视为未匹配
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(det)
            unmatched_trackers.append(trk)
        # 将过滤后的维度变成 1*2 形式
        # 比如，m 是 [0,0]，那么通过 reshape 会变换成 [[0,0]]，[1,1] 会变成 [[1,1]]
        # 那么 matches 就会变成 [array([[0,0]]), array([[1,1]])]
        else:
            matches.append(np.r_[det, trk].reshape(1, 2))

    # 如果过滤后匹配将结果为空，那么返回空的匹配结果
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=object)
    # 如果过滤后的匹配结果非空，那么按 0 轴方向继续添加匹配对
    # numpy 提供了 numpy.concatenate((a1,a2,...), axis=0) 函数。能够一次完成多个数组的拼接。其中 a1,a2,... 是数组类型的参数
    # 比如，a = np.array([[1,2,3], [4,5,6]]), b = np.array([[11, 21, 31], [7,8,9]])
    # np.concatenate((a,b), axis=0) 结果为 array([[1,2,3], [4,5,6], [11, 21, 31], [7, 8, 9]])
    # 那么前面的 matches 就会变成 array([[0,0], [1,1]])
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)