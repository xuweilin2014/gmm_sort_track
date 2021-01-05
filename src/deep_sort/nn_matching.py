# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """
    Compute pair-wise squared distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """

    # 用于计算成对的平方距离
    # a NxM 代表 N 个对象，每个对象有 M 个数值作为 embedding 进行比较
    # b LxM 代表 L 个对象，每个对象有 M 个数值作为 embedding 进行比较
    # 返回的是 NxL 的矩阵，比如 dist[i][j] 代表 a[i] 和 b[j] 之间的平方和距离
    # 实现见：https://blog.csdn.net/frankzd/article/details/80251042
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """
    Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to length 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """

    # a 和 b 之间的余弦距离
    # a : [NxM] b : [LxM]
    # 余弦距离 = 1 - 余弦相似度
    # https://blog.csdn.net/u013749540/article/details/51813922
    if not data_is_normalized:
        # 需要将余弦相似度转化成类似欧氏距离的余弦距离。
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        #  np.linalg.norm 操作是求向量的范式，默认是L2范式，等同于求向量的欧式距离
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """
    Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """

    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """

    # 假设 x 的大小为 S*M，y 的大小为 N*M
    # x 代表某个 track 中的 s 个 feature 向量，每个向量的维度为 M
    # y 代表 N 个 detection，每个 detection 有一个维度为 M 的 feature 向量
    # distances 是一个 S*N 矩阵，distances[i][j] 表示一个 track 的第 i 个 feature 向量和第 j 个 detection 的 feature 之间的余弦距离
    distances = _cosine_distance(x, y)

    # distances.min(axis=0) 返回一个长度为 1*N 的向量，代表了这个 track 的 feature 列表到每一个 detection 的 feature 的余弦距离的最小值
    return distances.min(axis=0)


# 对于每一个目标，返回一个最近距离
class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        # 默认 matching_threshold = 0.2，budge = 100
        if metric == "euclidean":
            # 使用最近邻欧式距离
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            # 使用最近邻的余弦距离
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        # 默认为 0.2
        self.matching_threshold = matching_threshold
        # 默认为 100
        self.budget = budget
        # samples是一个字典{id->feature list}
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        Update the distance metric with new data.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        # features 是一个 NxM 的矩阵，表示有 N 个维度为 M 的 feature 向量
        for feature, target in zip(features, targets):
            # Python 字典 setdefault() 函数和 get() 方法 类似, 如果键不存在于字典中，将会添加键并将值设为默认值
            self.samples.setdefault(target, []).append(feature)
            # 如果 budget 不为空 ，那么只取每个 track 的最新的 budget 个向量，budget 的值默认为 100，也就是只保存最新的 budget 个 feature
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        # 只保留 samples 中处于 confirmed 状态的 track 的 feature 向量
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """
        Compute distance between features and targets.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """

        # cost_matrix 的大小为 N * M，其中 N 为待匹配的 track 的个数，M 为 unmatched_detections 的个数
        # 最后计算出来的 cost_matrix[i][j] 代表 track[i] 的所有特征向量（feature）到 detection[j] 的 feature 属性的最小值
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            # self.samples 存储了所有的 track 中的最近 100 个特征向量（feature），而 samples[target] 则代表 track_id 为 target 的 feature 列表
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
