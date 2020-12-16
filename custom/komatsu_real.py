import os
import glob
from custom.tools import transform
from custom.label import color2label_komatsu_real

dataset_dir = 'dataset/komatsu_real'
rgb_dir = os.path.join(dataset_dir, 'imageCompressed')
seg_dir = os.path.join(dataset_dir, 'segmentation')
label_dir = os.path.join(dataset_dir, 'label')
list_dir = os.path.join(dataset_dir, 'list')


def makeList():
    label_paths = glob.glob(os.path.join(label_dir, '*'))
    label_paths.sort()
    label_paths = [path.replace(dataset_dir + '/', '') for path in label_paths]
    rgb_paths = glob.glob(os.path.join(rgb_dir, '*'))
    rgb_paths.sort()
    rgb_paths = [path.replace(dataset_dir + '/', '') for path in rgb_paths]

    open(os.path.join(list_dir, 'val.txt'), 'w').writelines(
        [' '.join([rgb, label]) + '\n' for rgb, label in zip(rgb_paths, label_paths)])


if __name__ == '__main__':
    # transform(label_dir, seg_dir, color2label_komatsu_real)
    makeList()
