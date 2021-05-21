import cv2
from time import sleep
from old_sort.sort import Sort
import numpy as np
import time

class Detector(object):

    def __init__(self, name='my_video', color=(0, 255, 0)):

        self.name = name
        self.color = color

    # noinspection PyAttributeOutsideInit
    def catch_video(self, video_index, min_area):

        cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类
        tracker = Sort()

        if not cap.isOpened():
            # 如果没有检测到摄像头，报错
            raise Exception('Check if the camera is on.')

        frame_count = -1
        COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

        self.frame_num = 0
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        begin_time = time.time()

        while cap.isOpened():
            sleep(0.1)

            catch, frame = cap.read()  # 读取每一帧图片

            if not catch:
                print('The end of the video.')
                break

            mask, frame = self.gaussian_bk(frame)

            frame_count += 1

            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
            dets = []

            for b in bounds:
                x, y, w, h = b
                dets.append([x, y, x + w, y + h, 1])

            dets = np.asarray(dets)
            ret, tracks, trks = tracker.update(dets)

            boxes = []
            indexIDs = []

            for bbox in ret:
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                indexIDs.append(int(bbox[4]))

            if len(boxes) > 0:
                for id, box in zip(indexIDs, boxes):
                    (left_x, top_y) = (int(box[0]), int(box[1]))
                    (right_x, bottom_y) = (int(box[2]), int(box[3]))
                    color = [int(c) for c in COLORS[id % len(COLORS)]]

                    dup = False

                    for track in tracks:
                        if track.id == id and not track.print_path:
                            dup = True
                            track.print_path = True
                            boxes = track.path
                            for i in range(len(boxes)):
                                print('{0},{1},{2:.0f},{3:.0f},{4:.0f},{5:.0f}'.format(
                                    frame_count - len(boxes) + i + 1, id, boxes[i][0],
                                    boxes[i][1],  boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]
                                ))

                    if not dup:
                        print('{0},{1},{2:.0f},{3:.0f},{4:.0f},{5:.0f}'.format(
                            frame_count, id, box[0],
                            box[1], box[2] - box[0], box[3] - box[1]
                        ))

                    cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), color, 3)
                    cv2.putText(frame, "{}".format(id), (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            for track in tracks:
                if track.downward():
                    dets = track.path
                    for i in range(len(dets)):
                        if i > 0:
                            color = [int(c) for c in COLORS[(track.id + 1) % len(COLORS)]]
                            cv2.line(frame, (self.get_center(dets[i])), (self.get_center(dets[i - 1])), color, 3)

            cv2.imwrite('../../output/trajectory/old/' + str(frame_count) + '.jpg', frame)
            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", 600, 800)
            cv2.imshow("result", frame)
            cv2.waitKey(10)

        end_time = time.time()
        print("fps: " + str((cap.get(7) / (end_time - begin_time))))

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
    detector = Detector(name='test')
    detector.catch_video('../../input/p.mp4', min_area=500)
