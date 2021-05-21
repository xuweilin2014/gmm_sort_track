# 单目标跟踪的评价指标
import numpy as np
import matplotlib.pyplot as plt
import os

# precision plot, 估计的目标位置（bounding box）的中心点与人工标注（ground-truth）的目标的中心点，
# 这两者的距离小于给定阈值的视频帧的百分比。
def compute_ope(dirs, mot_path, groundtruth_path):
    vor = {}
    dists = np.linspace(0, 50, 101, dtype=float)

    for dir in dirs:
        vot_dir_path = vot_path + dir + '/'
        vor_per = {}
        vor[dir] = vor_per

        vot_ids = str_list_to_num(os.listdir(vot_dir_path))
        gt_base_path = '../../../output/vot_metrics/groundtruth/'
        gt_ids = os.listdir(gt_base_path)
        gt_ids.sort()
        counter = 1

        for i in range(len(vot_ids)):
            id = vot_ids[i]
            gt_id = gt_ids[i]
            gt_dict = {}

            with open(gt_base_path + gt_id, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    box = [int(_) for _ in line.split(',')]
                    gt_dict[box[0]] = box[2:6]

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
                vot_box = np.array(vot_dict[frame_count])

                dist = get_dist(gt_box, vot_box)

                counter += 1
                for d in dists:
                    vor_per.setdefault(d, 0)
                    if dist <= d:
                        vor_per[d] += 1

        x = dists
        y = [_ / (float(547)) for _ in vor_per.values()]
        plt.title('Precision plots of OPE')
        label = dir.replace('_', ' ')

        plt.plot(x, y, linewidth=2, label=label)

    plt.legend()
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.grid()
    plt.savefig("ope.png", dpi=900, bbox_inches='tight')
    plt.show()

# 追踪算法得到的 bounding box（记为 a），与 ground-truth 给的 box（记为b），重合率定义为：OS = |a∩b|/|a∪b|，|·|表示区域的像素数目。
# 当某一帧的 OS 大于设定的阈值时，则该帧被视为成功的（Success），总的成功的帧占所有帧的百分比即为成功率（Success rate）
def compute_vor(dirs, vot_path, groundtruth_path):
    vor = {}
    ratios = np.linspace(0, 1, 101, dtype=float)

    for dir in dirs:
        vot_dir_path = vot_path + dir + '/'
        vor_per = {}
        vor[dir] = vor_per

        vot_ids = str_list_to_num(os.listdir(vot_dir_path))
        gt_base_path = '../../../output/vot_metrics/groundtruth/'
        gt_ids = os.listdir(gt_base_path)
        gt_ids.sort()
        counter = 1

        for i in range(len(vot_ids)):
            id = vot_ids[i]
            gt_id = gt_ids[i]
            gt_dict = {}

            with open(gt_base_path + gt_id, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    box = [int(_) for _ in line.split(',')]
                    gt_dict[box[0]] = box[2:6]

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
        y = [_ / (float(547)) for _ in vor_per.values()]
        plt.title('Success plots of OPE')
        label = dir.replace('_', ' ')

        plt.plot(x, y, linewidth=2, label=label)

    plt.legend()
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.grid()
    plt.savefig("vor.png", dpi=900, bbox_inches='tight')
    plt.show()


def str_list_to_num(ids):
    ids = [int(_.replace('.txt', '')) for _ in ids]
    ids.sort()
    return [str(id) + '.txt' for id in ids]

def get_dist(gt_box, vot_box):
    gt_center = np.array([gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2])
    vot_center = np.array([vot_box[0] + vot_box[2] / 2, vot_box[1] + vot_box[3] / 2])
    dist = np.linalg.norm(gt_center - vot_center)
    return dist

def iou(rec1, rec2):
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)


if __name__ == '__main__':
    dirs = ['cn', 'fhog', 'fhog_cn', 'old', 'raw_pixel']
    mot_path = '../../../output/mot_metrics/'
    vot_path = '../../../output/vot_metrics/'
    groundtruth_path = '../../../output/groundtruth/gt.txt'
    compute_ope(dirs, mot_path, groundtruth_path)
    compute_vor(dirs, vot_path, groundtruth_path)
