from tracker import DeepSortTracker
import nn_matching
from detection import Detection
from utils import *


# noinspection PyAttributeOutsideInit
class Runner(object):

    def __init__(self, color=(0, 255, 0), max_cosine_distance=0.3, nn_budget=100, distance_type="cosine", min_area=150):
        self.color = color
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.distance_type = distance_type

        metric = nn_matching.NearestNeighborDistanceMetric(self.distance_type, self.max_cosine_distance, self.nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        self.cell_size = 4
        self.tmpl_sz = [0, 0]
        self.padding = 2
        self.template_size = 96
        self._roi = [0., 0., 0., 0.]

        self.min_area = min_area

    @staticmethod
    def extract_image_patch(image, bbox, patch_shape=np.array([64, 48])):
        # bbox 的格式为 x y width height，也就是 yolo 检测框在 image 图片上的坐标
        bbox = np.array(bbox)
        if patch_shape is not None:
            # target_aspect 为 width / height，也就是说 patch_shape 的格式为（height, width)
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            # 根据 bbox 的高度算出符合 target_aspect 的 new_width，新宽度
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # 将 bbox 由 x y width height 转变为 top_left.x top_left.y bottom_right.x bottom_right.y
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        # 保证 bbox 框都在 image 的长宽范围之内
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox

        # 根据 bbox 的坐标在 image 上进行裁减
        image = image[sy:ey, sx:ex]
        # 将裁减好的 image 的长宽高转变为 patch_shape
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def get_feature(self, image, bbox):
        image = self.extract_image_patch(image, bbox)
        hog_feature, size_patch = extract_hog_feature(image, self.cell_size)
        cn_feature = extract_cn_feature(image, self.cell_size)
        feature = np.concatenate((hog_feature, cn_feature), axis=0)
        self.size_patch = list(map(int, [size_patch[0], size_patch[1], size_patch[2] + 11]))
        self.createHanningMats()
        feature = self.hann * feature

        shape = feature.shape
        return feature.reshape(shape[0] * shape[1])

    # 初始化 hanning 窗口，函数只在第一帧被执行
    # 目的是采样时为不同的样本分配不同的权重，0.5 * 0.5 是用汉宁窗归一化为 [0, 1]，得到的矩阵值就是每个样本的权重
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
        self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        self.hann = self.hann.astype(np.float32)

    # # 获取到目标的特征向量，使用了两种特征，fhog 特征以及 cn 颜色特征，并且对获取到的这两种特征进行了一个融合
    # def get_feature(self, image, bound):
    #     # self._roi 表示初始的目标框 [x, y, width, height]
    #     extracted_roi = [0, 0, 0, 0]
    #     # cx, cy 表示目标框中心点的 x 坐标和 y 坐标
    #     cx = bound[0] + bound[2] / 2  # float
    #     cy = bound[1] + bound[3] / 2  # float
    #
    #     # 保持初始目标框中心不变，将目标框的宽和高同时扩大相同倍数
    #     padded_w = bound[2] * self.padding
    #     padded_h = bound[3] * self.padding
    #
    #     # 设定模板图像尺寸为 96，计算扩展框与模板图像尺寸的比例
    #     # 把最大的边缩小到 96，_scale 是缩小比例，_tmpl_sz 是滤波模板裁剪下来的 PATCH 大小
    #     # scale = max(w,h) / template
    #     _scale = max(padded_h, padded_w) / float(self.template_size)
    #     # 同时将 scale 应用于宽和高，获取图像提取区域
    #     # roi_w_h = (w / scale, h / scale)
    #     self.tmpl_sz[0] = int(padded_w / _scale)
    #     self.tmpl_sz[1] = int(padded_h / _scale)
    #
    #     # 由于后面提取 hog 特征时会以 cell 单元的形式提取，另外由于需要将频域直流分量移动到图像中心，因此需保证图像大小为 cell大小的偶数倍，
    #     # 另外，在 hog 特征的降维的过程中是忽略边界 cell 的，所以还要再加上两倍的 cell 大小
    #     self.tmpl_sz[0] = int(self.tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
    #     self.tmpl_sz[1] = int(self.tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
    #
    #     # 选取从原图中扣下的图片位置大小
    #     extracted_roi[2] = int(_scale * self.tmpl_sz[0])
    #     extracted_roi[3] = int(_scale * self.tmpl_sz[1])
    #     extracted_roi[0] = int(cx - extracted_roi[2] / 2)
    #     extracted_roi[1] = int(cy - extracted_roi[3] / 2)
    #
    #     # z 是当前被裁剪下来的搜索区域
    #     z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
    #     if z.shape[1] != self.tmpl_sz[0] or z.shape[0] != self.tmpl_sz[1]:
    #         z = cv2.resize(z, tuple(self.tmpl_sz))
    #
    #     # 获取到 fhog 特征，也就是梯度直方图特征
    #     hog_feature, size_patch = extract_hog_feature(z, self.cell_size)
    #     # 获取到 cn 颜色特征
    #     cn_feature = extract_cn_feature(z, self.cell_size)
    #     FeaturesMap = np.concatenate((hog_feature, cn_feature), axis=0)
    #
    #     # size_patch 为列表，保存裁剪下来的特征图的 [长，宽，通道]
    #     self.size_patch = list(map(int, [size_patch[0], size_patch[1], size_patch[2] + 11]))
    #     # create Hanning Mats need size_patch
    #     self.createHanningMats()
    #     # 加汉宁窗减少频谱泄漏
    #     FeaturesMap = self.hann * FeaturesMap
    #
    #     return FeaturesMap

    # noinspection PyAttributeOutsideInit
    def catch_video(self, video_index):
        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_count = 0

        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc_frame = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc_mask = cv2.VideoWriter.fourcc(*'MJPG')
        out_frame = cv2.VideoWriter('../output/result_frame.avi', fourcc_frame, 15, (w, h))
        out_mask = cv2.VideoWriter('../output/result_mask.avi', fourcc_mask, 15, (w, h))

        self.frame_num = 0

        while cap.isOpened():

            mask, frame = self.gaussian_bk(cap)
            frame_count += 1

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > self.min_area]
            dets = []

            for bound in bounds:
                x, y, w, h = bound
                cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 3)
                feature = self.get_feature(frame, bound)
                dets.append(Detection(np.array([x, y, w, h]), 1, feature))

            dets = np.asarray(dets)
            self.tracker.predict()
            self.tracker.update(dets)
            trks = self.tracker.tracks

            boxes = []
            indexIDs = []

            for trk in trks:
                box = trk.to_tlbr()
                boxes.append(box)
                indexIDs.append(trk.track_id)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)

            if len(boxes) > 0:
                for id, box in zip(indexIDs, boxes):
                    (left_x, top_y) = (int(box[0]), int(box[1]))
                    (right_x, bottom_y) = (int(box[2]), int(box[3]))

                    color = [int(c) for c in self.COLORS[id % len(self.COLORS)]]
                    cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), color, 3)
                    cv2.putText(frame, "{}".format(id), (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # for track in tracks:
            #     if track.downward():
            #         bboxes = track.path
            #         for i in range(len(bboxes)):
            #             if i > 0:
            #                 color = [int(c) for c in self.COLORS[(track.track_id) % len(self.COLORS)]]
            #                 cv2.line(frame, (self.get_center(bboxes[i])), (self.get_center(bboxes[i - 1])), color, 3)

            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 1200, 1000)
            cv2.imshow("result", frame)

            mask = np.expand_dims(mask, 2).repeat(3, axis=2)

            cv2.imshow("mask", mask)
            out_frame.write(frame)
            out_mask.write(np.expand_dims(mask, 2).repeat(3, axis=2))

            cv2.waitKey(10)

        # 释放摄像头
        cap.release()
        out_mask.release()
        out_frame.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_center(det):
        if det is not None:
            return int((det[0] + det[2]) / 2), int((det[1] + det[3]) / 2)
        return None

    def gaussian_bk(self, cap, k_size=3):
        catch, frame = cap.read()  # 读取每一帧图片

        if not catch:
            print('The end of the video.')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.mog.apply(gray).astype('uint8')
        mask = cv2.medianBlur(mask, k_size)

        return mask, frame


if __name__ == "__main__":
    runner = Runner()
    runner.catch_video('../input/p.mp4')
