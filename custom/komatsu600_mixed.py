import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from custom.label import color2label_komatsu600_mixed, rgb2label

variation = ['noon', 'evening']

dataset_dir = 'dataset/komatsu600_mixed'
segmentation_dir = os.path.join(dataset_dir, 'segmentation')
label_dir = os.path.join(dataset_dir, 'label')
depth_dir = os.path.join(dataset_dir, 'depth_cm16bit')
normalized_depth_dir = os.path.join(dataset_dir, 'normalized_3C_depth')
list_dir = os.path.join(dataset_dir, 'list')
list_depth_dir = os.path.join(dataset_dir, 'list-depth')


def transform():
    """
    transform rgb-segmentation label to train-id label
    """
    variation_dirs = [os.path.join(label_dir, v) for v in variation]
    for directory in variation_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    segmentation_paths = [path for path in glob.glob(os.path.join(segmentation_dir, '*', '*')) if
                          any([v in path for v in variation])]
    for path in tqdm(segmentation_paths):
        label = cv2.imread(path)[:, :, ::-1]
        new_label = rgb2label(label, color2label_komatsu600_mixed)
        cv2.imwrite(path.replace('segmentation', 'label'), new_label)


def normalize_depth():
    variation_dirs = [os.path.join(normalized_depth_dir, v) for v in variation]
    for directory in variation_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    depth_paths = [path for path in glob.glob(os.path.join(depth_dir, '*', '*')) if
                   any([v in path for v in variation])]
    for path in tqdm(depth_paths):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        normalized_depth = (depth / 65535 * 255).astype('uint8')
        normalized_3C_depth = np.stack((normalized_depth,) * 3, axis=-1)
        cv2.imwrite(path.replace('depth_cm16bit', 'normalized_3C_depth'), normalized_3C_depth)


def makeList_depth_random(train_rate=0.8):
    """
    Generate `list-depth/train.txt`, `list-depth/val.txt` files
    """

    def write(basenames, split):
        assert split in ['train', 'val']
        with open(os.path.join(list_depth_dir, f'{split}.txt'), 'w') as f:
            for basename in basenames[:train_num]:
                for v in variation:
                    rgb_path = os.path.join('rgb', v, basename)
                    depth_path = os.path.join('normalized_3C_depth', v, basename)
                    label_path = os.path.join('label', v, basename)
                    f.write(' '.join([rgb_path, depth_path, label_path]) + '\n')

    if not os.path.exists(list_depth_dir):
        os.mkdir(list_depth_dir)
    paths = sorted(glob.glob(os.path.join(label_dir, variation[0], '*')))
    basenames = [os.path.basename(p) for p in paths]
    total_num = len(basenames)
    train_num = int(total_num * train_rate)
    np.random.seed(0)
    np.random.shuffle(basenames)
    train_basenames = basenames[:train_num]
    val_basenames = basenames[train_num:]
    write(train_basenames, 'train')
    write(val_basenames, 'val')


def makeList_random():
    """
    Generator `list/train.txt`, `list/val.txt` files by 'list-depth/train.txt`, `list-depth/val.txt`
    """
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)
    paths = glob.glob(os.path.join(list_depth_dir, '*'))
    for path in paths:
        lines = open(path, 'r').readlines()
        new_lines = []
        for line in lines:
            splited_line = line.split(' ')
            new_line = ' '.join([splited_line[0], splited_line[2]])
            new_lines.append(new_line)
        open(path.replace('list-depth', 'list'), 'w').writelines(new_lines)


if __name__ == '__main__':
    makeList_random()
