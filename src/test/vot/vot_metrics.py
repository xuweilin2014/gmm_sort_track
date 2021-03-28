# 单目标跟踪的评价指标
import numpy as np
import matplotlib.pyplot as plt
from kcf.associate import iou
import os

# precision plot, 估计的目标位置（bounding box）的中心点与人工标注（ground-truth）的目标的中心点，
# 这两者的距离小于给定阈值的视频帧的百分比。
def compute_ope(dirs, mot_path, groundtruth_path):
    gt_dict = {}
    with open(groundtruth_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = [int(_) for _ in line.split(',')]
            gt_dict[box[0]] = box[:6]

    ape = {}
    dists = np.linspace(0, 50, 51, dtype=int)

    for dir in dirs:
        vot_file_path = mot_path + dir + '/' + dir + '.txt'
        vot_dict = {}
        ape_per = {}
        ape[dir] = ape_per

        with open(vot_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '视频' in line:
                    continue

                box = [int(_) for _ in line.split(',')]
                vot_dict[box[0]] = box[:6]

        for frame_count in gt_dict:
            gt_box = gt_dict[frame_count]

            if frame_count not in vot_dict:
                continue

            gt_center = np.array([gt_box[2] + gt_box[4] / 2, gt_box[3] + gt_box[5] / 2])
            vot_box = vot_dict[frame_count]
            vot_center = np.array([vot_box[2] + vot_box[4] / 2, vot_box[3] + vot_box[5] / 2])

            dist = np.linalg.norm(gt_center - vot_center)

            for d in dists:
                ape_per.setdefault(d, 0)
                if dist <= d:
                    ape_per[d] += 1

        dists = np.linspace(0, 50, 51, dtype=int)
        y = [_ / len(gt_dict) for _ in ape_per.values()]
        plt.title('Precision plots of OPE')

        label = dir.replace('_', ' ')

        plt.plot(dists, y, label=label, linewidth=3)

    plt.legend()
    plt.xlabel('Location error threshold')
    plt.ylabel('Success rate')
    plt.grid()
    plt.savefig("ope.png", dpi=900, bbox_inches='tight')
    plt.show()

# 追踪算法得到的 bounding box（记为 a），与 ground-truth 给的 box（记为b），重合率定义为：OS = |a∩b|/|a∪b|，|·|表示区域的像素数目。
# 当某一帧的 OS 大于设定的阈值时，则该帧被视为成功的（Success），总的成功的帧占所有帧的百分比即为成功率（Success rate）
def compute_vor(dirs, vot_path, groundtruth_path):
    gt_dict = {}
    with open(groundtruth_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = [int(_) for _ in line.split(',')]
            gt_dict[box[0]] = box[2:6]

    vor = {}
    ratios = np.linspace(0, 1, 101, dtype=float)

    for dir in dirs:
        vot_dir_path = vot_path + dir + '/'
        vor_per = {}
        vor[dir] = vor_per

        ids = os.listdir(vot_dir_path)
        counter = 1
        for id in ids:
            vot_file_path = vot_dir_path + id
            vot_dict = {}

            with open(vot_file_path, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    if '视频' in line:
                        continue

                    line = line.strip()
                    if len(line) == 0:
                        continue

                    box = [int(_) for _ in line.split(',')]
                    vot_dict[box[0]] = box[2:6]

            for frame_count in gt_dict:
                if frame_count not in vot_dict:
                    continue

                gt_box = np.array(gt_dict[frame_count])
                gt_box[2:] = gt_box[:2] + gt_box[2:]
                vot_box = np.array(vot_dict[frame_count])
                vot_box[2:] = vot_box[:2] + vot_box[2:]

                o = iou(vot_box, gt_box)
                print(frame_count, o)
                counter += 1
                for r in ratios:
                    vor_per.setdefault(r, 0)
                    if o < 0:
                        vor_per.setdefault(0, 0)
                        vor_per[0] += 1
                        continue
                    if o >= r:
                        vor_per[r] += 1

        x = ratios
        y = [_ / (float(len(gt_dict))) for _ in vor_per.values()]
        plt.title('Success plots of OPE')
        label = dir.replace('_', ' ')

        plt.plot(x, y, linewidth=2, label=label)

    plt.legend()
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.grid()
    plt.savefig("vor.png", dpi=900, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # dirs = ['cn', 'fhog', 'fhog_cn', 'old', 'raw_pixel']
    dirs = ['old']
    mot_path = '../../../output/mot_metrics/'
    vot_path = '../../../output/vot_metrics/'
    groundtruth_path = '../../../output/groundtruth/gt.txt'
    # compute_ope(dirs, mot_path, groundtruth_path)
    compute_vor(dirs, vot_path, groundtruth_path)
