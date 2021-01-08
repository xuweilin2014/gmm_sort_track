import numpy as np
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment


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

def kcf_associate(frame, unmatched_dets, unmatched_trks, dets, trks):
    unmatched_dets_final = []
    unmatched_trks_final = []
    peak_threshold = 0.1
    matched = []

    for trk_index in unmatched_trks:
        distance = []
        trk = trks[trk_index]
        box, peak_value = trk.update_kcf(frame)

        if peak_threshold > peak_value:
            continue

        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        for det_index in unmatched_dets:
            det = dets[det_index]
            det_box = det.to_tlwh()
            det_cx = det_box[0] + det_box[2] / 2
            det_cy = det_box[1] + det_box[3] / 2
            dist = (cx - det_cx) ** 2 + (cy - det_cy) ** 2
            if dist < max(box[2], box[3]) * 4:
                distance.append([(cx - det_cx) ** 2 + (cy - det_cy) ** 2, det_index])

        distance = np.array(distance)
        if len(distance) > 0 and len(dets) > 0:
            min_index = int(distance.min(axis=0)[1])
            trk.retrain_kcf(frame, dets[min_index].to_tlwh())
            matched.append([min_index, trk_index])

    matched = np.array(matched)
    for det_index in unmatched_dets:
        if len(matched) == 0 or det_index not in matched[:, 0]:
            unmatched_dets_final.append(det_index)

    for trk_index in unmatched_trks:
        if len(matched) == 0 or trk_index not in matched[:, 1]:
            unmatched_trks_final.append(trk_index)

    if len(matched) == 0:
        matched = np.empty((0, 2), dtype=int)

    return matched, unmatched_dets_final, unmatched_trks_final


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1):
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