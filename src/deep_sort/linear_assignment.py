# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    Solve linear assignment problem.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # 1.iou 匹配
    # distance_metric 方法返回一个 cost_matrix，这个 cost_matrix 的大小为 N*M，其中 N 是 len(track_indices)，也就是 len(unmatched_tracks)
    # M 是指 len(detection_indices)，也就是 len(unmatched_detections)。cost_matrix[i][j] 也就是第 i 个 track 的目标框和第 j 个 detection 的目标框的
    # 1 - iou(track[i],detection[j])，也就是 cost_matrix[i][j] 的值越大，说明 track[i] 和 detection[j] 之间的 iou 越小，不大可能匹配
    #
    # 2.外观特征匹配和运动信息匹配（gated_metric)
    # 先用外观特征计算出一个 cost_matrix，这个 cost_matrix 表示每个 track 和 detection 的 feature 之间的余弦距离，
    # 再使用马氏距离对这个 cost_matrix 进行门限控制，最后返回
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)

    # 把 cost_matrix 中大于阈值 max_distance （默认为 0.7）的值设置为 0.70001
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # 使用匈牙利算法，得到一个 N * 2 的结果，也就是 N 个匹配的 (track, detection）
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # 获取到经过匈牙利算法还是没有匹配的 detection 的索引
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    # 获取到经过匈牙利算法还是没有匹配到的 track 的索引
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    # 注意，对于已经在 indices 中的 track 和 detection，如果它们在 cost_matrix 中的值大于阈值 max_distance，那么此 track 和此 detection 也认为是不匹配的
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    """
    Run matching cascade.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """

    # 如果 track_indices 或者 detection_indices 是 none，那就重新生成索引列表
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 最开始的 unmatched_detections 中包含了所有的 detection 索引，也就是初始状态所有的 detection 都看成是 unmatched_detection
    unmatched_detections = detection_indices
    matches = []

    # cascade_depth 的值等于 max_age，从 time_since_update = 0 的 track 开始和 detection 进行匹配
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        # detection 只和 time_since_update == level + 1 的帧进行匹配
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]

        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        # 在每一次迭代过程中，unmatched_detections 都可能会和部分 track 匹配成功从而使得数量减少
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)

        # 将每一次迭代过程中和 detection 进行匹配成功的 track id 和 detection id 保存起来
        matches += matches_l

    # 在所有的 track id 中减去已经匹配上的 track 的 id，就得到了 unmatched track 的 id 集合
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    """
    Invalidate infeasible entries in cost matrix based on the state distributions obtained by Kalman filtering.
    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """

    # cost_matrix 的大小为 N * M，表示 N 个 track 和 M 个 detection 之间的余弦距离，其中 N = len(track_indices), M = len(detection_indices)
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # 将 detection 的 bounding box 的格式改为（center_x, center_y, aspect ratio, height）
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])

    # 对于 track，计算其预测结果和检测结果之间的马氏距离，并且将 cost_matrix 中相应 track 的马氏距离大于阈值（gating threshold）
    # 的值置为 gated_cost（默认值为 10000），也就是无穷大。
    # 其实也就是将前面计算出来的代表余弦距离的 cost_matrix 再通过各个 track 和 detection 之间的马氏距离筛选一遍
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
