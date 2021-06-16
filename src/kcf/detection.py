import numpy as np


class Detection:
    def __init__(self, tlwh, confidence):
        # tlwh 形式为 [x, y, width, height]
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)

    # tlbr 的形式为 [x1, y1, x2, y2]，(x1,y1) 和 (x2,y2) 分别是检测框左上角和右下角的点坐标
    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # xyah 的形式为 [x, y, width / height, height]
    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[2] /= ret[3]
        return ret

    def to_tlwh(self):
        return self.tlwh.copy()