import cv2
import numpy as np
from src.gmm_sort import Detection
from src.gmm_sort import Sort


class Detector(object):

    def __init__(self, name='my_video', frame_num=10, k_size=7, color=(0, 255, 0)):

        self.name = name
        self.color = color

    # noinspection PyAttributeOutsideInit
    def catch_video(self, video_index, min_area):
        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类
        tracker = Sort()

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_count = 0

        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc_frame = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc_mask = cv2.VideoWriter.fourcc(*'MJPG')
        out_frame = cv2.VideoWriter('./output/result_frame.avi', fourcc_frame, 15, (w, h))
        out_mask = cv2.VideoWriter('./output/result_mask.avi', fourcc_mask, 15, (w, h))

        COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

        self.frame_num = 0
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        while cap.isOpened():

            mask, frame = self.gaussian_bk(cap)

            frame_count += 1

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
            dets = []

            for b in bounds:
                x, y, w, h = b
                cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 3)
                dets.append(Detection(np.array([x, y, w, h]), 1))

            dets = np.asarray(dets)
            tracker.predict()
            ret, tracks, trks = tracker.update(dets)

            boxes = []
            indexIDs = []

            for trk in trks:
                cv2.rectangle(frame, (int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])), (0, 0, 255), 3)

            for bbox in ret:
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                indexIDs.append(int(bbox[4]))

            if len(boxes) > 0:
                for id, box in zip(indexIDs, boxes):
                    (left_x, top_y) = (int(box[0]), int(box[1]))
                    (right_x, bottom_y) = (int(box[2]), int(box[3]))

                    color = [int(c) for c in COLORS[id % len(COLORS)]]
                    cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), color, 3)
                    cv2.putText(frame, "{}".format(id), (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            for track in tracks:
                if track.downward():
                    bboxes = track.path
                    for i in range(len(bboxes)):
                        if i > 0:
                            color = [int(c) for c in COLORS[(track.track_id) % len(COLORS)]]
                            cv2.line(frame, (self.get_center(bboxes[i])), (self.get_center(bboxes[i - 1])), color, 3)

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
    detector = Detector(name='test')
    detector.catch_video('./input/p.mp4', min_area=150)
