import time
import cv2
import numpy as np
import logging
from detection import Detection
from sort import Sort
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

class Detector(object):

    def __init__(self, name='my_video', cn=True, hog=True, color=(0, 255, 0)):

        self.name = name
        self.color = color
        self.hog = hog
        self.cn = cn

    # noinspection PyAttributeOutsideInit
    def catch_video(self, video_index, min_area):
        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类
        tracker = Sort(cn=self.cn, hog=self.hog)

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_count = -1

        self.frame_num = 0
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        begin_time = time.time()
        obj_dict = {}

        while cap.isOpened():
            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:
                print('The end of the video.')
                break

            mask, frame = self.gaussian_bk(frame)
            frame_count += 1

            if 760 <= frame_count < 965:
                continue

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
            dets = []

            for b in bounds:
                x, y, w, h = b
                dets.append(Detection(np.array([x, y, w, h]), 1))

            dets = np.asarray(dets)
            tracker.predict()
            ret, tracks = tracker.update(frame, frame_count, dets)

            boxes = []
            indexIDs = []

            for bbox in ret:
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                indexIDs.append(int(bbox[4]))

            if len(boxes) > 0:
                for id, box in zip(indexIDs, boxes):
                    if id != 252 and id != 398:
                        (left_x, top_y) = (int(box[0]), int(box[1]))
                        (right_x, bottom_y) = (int(box[2]), int(box[3]))

                        color = [int(c) for c in COLORS[id % len(COLORS)]]

                        obj_dict.setdefault(frame_count, [])
                        dup = False

                        for track in tracks:
                            if track.track_id == id and not track.print_path:
                                dup = True
                                track.print_path = True
                                tboxes = track.path
                                for i in range(len(tboxes)):
                                    obj_dict.setdefault(tboxes[i][0], [])
                                    obj_dict.get(tboxes[i][0]).append([frame_count - (len(tboxes)) + i + 1, id, int(tboxes[i][1]),
                                                                            int(tboxes[i][2]), int(tboxes[i][3] - tboxes[i][1]), int(tboxes[i][4] - tboxes[i][2])])

                        if not dup:
                            obj_dict.get(frame_count).append([frame_count, id, int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])])

                        cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), color, 2)
                        cv2.putText(frame, "{}".format(id), (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            for track in tracks:
                id = track.track_id
                if id != 252 and id != 398:
                    bboxes = track.path
                    for i in range(len(bboxes)):
                        if i > 0:
                            color = [int(c) for c in COLORS[id % len(COLORS)]]
                            cv2.line(frame, (self.get_center(bboxes[i][1:])), (self.get_center(bboxes[i - 1][1:])), color, 3)

            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 600, 1000)
            cv2.imshow("result", frame)

            mask = np.expand_dims(mask, 2).repeat(3, axis=2)

            cv2.imshow("mask", mask)
            cv2.waitKey(10)

        end_time = time.time()
        fps = cap.get(7) / (end_time - begin_time)

        file_path = ''
        if self.hog and self.cn:
            file_path = '../../../output/samf/samf_test.txt'
        elif self.hog:
            file_path = '../../../output/fhog/hog_test.txt'
        elif self.cn:
            file_path = '../../../output/cn/cn_test.txt'
        elif not self.hog and not self.cn:
            file_path = '../../../output/raw_pixel/raw_pixel_test.txt'

        f = open(file_path, 'w')
        for frame_count in sorted(obj_dict):
            for path in obj_dict.get(frame_count):
                f.write(','.join([str(_) for _ in path]) + '\n')
        f.write('视频帧率：' + str(fps))
        f.close()
        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_center(det):
        if det is not None:
            return int((det[0] + det[2]) / 2), int((det[1] + det[3]) / 2)
        return None

    def gaussian_bk(self, frame, k_size=3):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.mog.apply(gray).astype('uint8')
        mask = cv2.medianBlur(mask, k_size)

        return mask, frame


if __name__ == "__main__":
    video_path = '../../../input/p.mp4'
    min_area = 130

    detector = Detector(name='test', hog=True, cn=False)
    detector.catch_video(video_path, min_area=min_area)
    logging.info('fhog 特征的 test 文件生成完毕')

    detector = Detector(name='test', hog=True, cn=True)
    detector.catch_video(video_path, min_area=min_area)
    logging.info('fhog 与 raw_pixel 颜色特征的 test 文件生成完毕')

    detector = Detector(name='test', hog=False, cn=False)
    detector.catch_video(video_path, min_area=min_area)
    logging.info('raw_pixel 普通灰度特征的 test 文件生成完毕')

    detector = Detector(name='test', hog=False, cn=True)
    detector.catch_video(video_path, min_area=min_area)
    logging.info('cn 颜色特征的 test 文件生成完毕')
