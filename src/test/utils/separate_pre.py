import os

# noinspection PyShadowingBuiltins
def separate_predictions(dirs, mot_path, vot_path):
    for dir in dirs:
        mot_dir_path = mot_path + dir
        vot_dir_path = vot_path + dir

        if os.path.exists(vot_dir_path):
            file_paths = os.listdir(vot_dir_path)
            for file_path in file_paths:
                os.remove(vot_dir_path + '/' + file_path)
        else:
            os.mkdir(vot_dir_path)

        mot_file_path = mot_dir_path + '/' + dir + '.txt'

        with open(mot_file_path, 'r') as mot_f:
            lines = mot_f.readlines()

            for line in lines:
                if '视频' in line:
                    continue

                box = [_ for _ in line.split(',')]
                id = box[1]
                vot_file_path = vot_dir_path + '/' + id + '.txt'

                with open(vot_file_path, 'a+') as vot_f:
                    vot_f.write(','.join(box[:6]) + '\n')


if __name__ == '__main__':
    dirs = ['cn', 'fhog', 'fhog_cn', 'old', 'raw_pixel']
    mot_path = '../../../output/mot_metrics/'
    vot_path = '../../../output/vot_metrics/'
    separate_predictions(dirs, mot_path, vot_path)
