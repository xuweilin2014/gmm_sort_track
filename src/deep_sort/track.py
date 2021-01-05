# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """

    # Tentative: 不确定态，这种状态会在初始化一个 Track 的时候分配，并且只有在连续匹配上 n_init 帧才会转变为确定态。如果在处于不确定态的情况下没有匹配上任何 detection，那将转变为删除态。
    # Confirmed: 确定态，代表该 Track 确实处于匹配状态。如果当前 Track 属于确定态，但是失配连续达到 max_age 次数的时候，就会被转变为删除态。
    # Deleted: 删除态，说明该 Track 已经失效。
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id

        # hits 是在每次 update 的时候加 1，也就是说只有在 track 和某个 detection 匹配上的时候才会加一
        # hits 代表了一共匹配上了多少次，当 hits > n_init 的时候，就会将 track 设置为 confirmed 状态
        self.hits = 1
        self.age = 1

        # time_since_update 在调用 predict 的时候会加 1，而每次调用 update 的时候（也就是匹配上的时候）置为 0，也就是说 time_since_update 表示失配多少次
        # 当 time_since_update 大于 max_age 的时候就会将 track 的状态设置为 deleted 状态
        self.time_since_update = 0

        self.state = TrackState.Tentative

        # features 列表，存储该轨迹在不同帧对应位置通过 ReID 提取到的特征
        self.features = []

        # 每个 track 都对应多个 feature，每次更新都将最新的 feature 添加到列表中
        if feature is not None:
            self.features.append(feature)

        # 一个新创建的 track 最开始连续 n_init 帧匹配上 detection 之后才会从 Tentative 状态转变为 Confirmed 状态
        # 如果最开始 n_init 帧中有任何一帧没有匹配上任何 detection，那么就会被标记为 deleted 状态
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        Returns
        -------
        ndarray
            The bounding box.
        """

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """
        Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        Returns
        -------
        ndarray
            The bounding box.
        """

        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """

        # 根据卡尔曼滤波算法，对当前帧的跟踪结果进行预测，预测完了之后，会对每一个 track 的 time_since_update 加 1
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """
        Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())

        # 将新增的 feature 增加到此 track 的 features 属性中
        self.features.append(detection.feature)
        # hits 代表了一共匹配上了多少次，这里也要加一
        self.hits += 1
        # 此 track 匹配上了一个 detection，将 time_since_update 重置为 0
        self.time_since_update = 0
        # 一个新创建的 track，当其 hits >= n_init 也就是连续匹配上 n_init 次，那么就将这个 track 的状态修改为 Confirmed 状态
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """
        Mark this track as missed (no association at the current time step).
        """
        # 若 track 处于未确定 Tentative 状态，则标记为 deleted 状态
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # 若 track 失配的次数大于 max_age，那么也标记为 deleted 状态
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """
        Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
