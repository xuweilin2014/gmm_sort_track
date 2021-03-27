import os

def merge_groundtruth():
    # 用来将 1-9 个物体的 groundtruth 文件合并成一个文件
    gt_path = '../../../output/groundtruth/gt.txt'
    if os.path.exists(gt_path):
        os.remove(gt_path)
    gt = open(gt_path, 'w')

    obj_dict = {}
    ext = '.txt'
    base_path = '../../../output/groundtruth/id_'
    for i in range(1, 10):
        file_path = base_path + str(i) + ext

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue

                nums = [int(_) for _ in line.split(',')]

                obj_dict.setdefault(nums[0], []).append(nums)

    with open(gt_path, 'w') as f:
        for frame_count in sorted(obj_dict.keys()):
            for bbox in obj_dict[frame_count]:
                new_line = ','.join([str(_) for _ in bbox]) + ',1,-1,-1\n'
                f.write(new_line)

def append_test_metrics(name_list):
    for path in name_list:
        path_list = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if '-1' not in line:
                    line = line + ',1,-1,-1'
                path_list.append(line + '\n')

        with open(path, 'w') as f:
            for p in path_list:
                f.write(p)


if __name__ == '__main__':
    names = ['../../../output/cn/cn_test.txt']
    append_test_metrics(names)
