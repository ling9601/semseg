import numpy as np
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os


class Label:
    def __init__(self, className, trainId):
        self.className = className
        self.trainId = trainId


color2label = {
    (210, 0, 200): Label('Terrain', 0),
    (90, 200, 255): Label('Sky', 1),
    (0, 199, 0): Label('Tree', 2),
    (90, 240, 0): Label('Vegetation', 3),
    (140, 140, 140): Label('Building', 4),
    (100, 60, 100): Label('Road', 5),
    (250, 100, 255): Label('GuardRail', 255),
    (255, 255, 0): Label('TrafficSign', 6),
    (200, 200, 0): Label('TrafficLight', 7),
    (255, 130, 0): Label('Pole', 8),
    (80, 80, 80): Label('Misc', 255),
    (160, 60, 60): Label('Truck', 9),
    (255, 127, 80): Label('Car', 10),
    (0, 139, 139): Label('Van', 11),
    (0, 0, 0): Label('Undefined', 255),
}


def rgb2label(img):
    """

    Args:
        img: CvImage in rgb order

    Returns:
        label: (H, W)
    """
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0], [1], [2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    label = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        label[img_id == c] = color2label[tuple(img[img_id == c][0])].trainId
    return label


def transform():
    pathes = glob.glob('../dataset/kitti2/label/*/*/*/*/*/*classgt_*.png')
    assert len(pathes) == 42520

    for path in tqdm(pathes):
        color = cv2.imread(path)[:, :, ::-1]
        label = rgb2label(color).astype('uint8')
        cv2.imwrite(path.replace('classgt', 'label'), label)


def rename():
    paths = glob.glob('../dataset/kitti2/*/*/*/*/*/*/*')
    for p in tqdm(paths):
        new_name = '+'.join(p.replace('../dataset/kitti2/', '').split('/'))
        os.rename(p, p.replace(os.path.basename(p), new_name))


def visualize():
    rgb_names = sorted(glob.glob('../dataset/kitti2/rgb/*/*/*/*/*/*/rgb_*.jpg'))
    color_names = sorted(glob.glob('../dataset/kitti2/label/*/*/*/*/*/*/classgt_*.png'))
    label_names = sorted(glob.glob('../dataset/kitti2/label/*/*/*/*/*/*/label_*.png'))
    rgb = cv2.imread(rgb_names[0])[:, :, ::-1]
    color = cv2.imread(color_names[0])[:, :, ::-1]
    label = cv2.imread(label_names[0], cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(rgb)
    axes[1].imshow(color)
    axes[2].imshow(label)
    plt.show()


def makeList_randomSplit(train_rate=0.9, variations=None, camera=0):
    """
    creat list 'train.txt', 'val.txt'
    Args:
        train_rate:
        variations: subset of [15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning, overcast, rain, sunset]
        camera: 0 or 1

    """
    if variations is None:
        variations = ['15-deg-left', '15-deg-right', 'clone']
    dst = '../dataset/kitti2/list'
    rgb_names = []
    label_names = []
    for i in variations:
        rgb_names.extend(glob.glob('../dataset/kitti2/rgb/*/{}/*/*/Camera_{}/*rgb_*.jpg'.format(i, camera)))
        label_names.extend(
            glob.glob('../dataset/kitti2/label/*/{}/*/*/Camera_{}/*label_*.png'.format(i, camera)))
    rgb_names.sort()
    label_names.sort()
    # only take one third of the data
    rgb_names = [n for n in rgb_names if int(n[-9:-4]) % 3 == 0]
    label_names = [n for n in label_names if int(n[-9:-4]) % 3 == 0]
    # remove data root path
    rgb_names = [i.replace('../dataset/kitti2/', '') for i in rgb_names]
    label_names = [i.replace('../dataset/kitti2/', '') for i in label_names]
    assert len(rgb_names) == len(label_names)
    # shuffle data
    idx = np.arange(len(rgb_names))
    np.random.seed(0)
    np.random.shuffle(idx)
    rgb_names = [rgb_names[i] for i in idx]
    label_names = [label_names[i] for i in idx]
    train_num = int(len(rgb_names) * train_rate)
    with open(os.path.join(dst, 'train.txt'), 'w') as f:
        f.writelines(
            [' '.join([rgb, label]) + '\n' for rgb, label in zip(rgb_names[:train_num], label_names[:train_num])])
    with open(os.path.join(dst, 'val.txt'), 'w') as f:
        f.writelines(
            [' '.join([rgb, label]) + '\n' for rgb, label in zip(rgb_names[train_num:], label_names[train_num:])])
    print('train({}), val({})'.format(train_num, len(rgb_names) - train_num))


def makeList(variation=None, camera=0, is_depth=False):
    if variation is None:
        variations = ['15-deg-left', '15-deg-right', 'clone']
    dst = '../dataset/kitti2/list'
    split = {
        'train': ['01', '06', '18', '20'],
        'val': ['02']
    }
    for section, scenes in split.items():
        rgb_names = []
        label_names = []
        if is_depth:
            depth_names = []
        for s in scenes:
            for v in variations:
                rgb_names.extend(
                    glob.glob('../dataset/kitti2/rgb/Scene{}/{}/*/*/Camera_{}/*rgb_*.jpg'.format(s, v, camera)))
                label_names.extend(
                    glob.glob('../dataset/kitti2/label/Scene{}/{}/*/*/Camera_{}/*label_*.png'.format(s, v, camera)))
                if is_depth:
                    depth_names.extend(
                        glob.glob(
                            '../dataset/kitti2/normalized_3C_depth/Scene{}/{}/*/*/Camera_{}/*depth_*.png'.format(s, v,
                                                                                                                 camera)))
        rgb_names.sort()
        label_names.sort()

        # only take one third of the data
        rgb_names = [n for n in rgb_names if int(n[-9:-4]) % 3 == 0]
        label_names = [n for n in label_names if int(n[-9:-4]) % 3 == 0]
        # remove data root path
        rgb_names = [i.replace('../dataset/kitti2/', '') for i in rgb_names]
        label_names = [i.replace('../dataset/kitti2/', '') for i in label_names]
        if is_depth:
            depth_names.sort()
            depth_names = [n for n in depth_names if int(n[-9:-4]) % 3 == 0]
            depth_names = [i.replace('../dataset/kitti2/', '') for i in depth_names]
            assert len(depth_names) == len(rgb_names)

        assert len(rgb_names) == len(label_names)
        with open(os.path.join(dst, '{}.txt'.format(section)), 'w') as f:
            if not is_depth:
                f.writelines(
                    [' '.join([rgb, label]) + '\n' for rgb, label in zip(rgb_names, label_names)]
                )
            else:
                f.writelines(
                    [' '.join([rgb, depth, label]) + '\n' for rgb, depth, label in zip(rgb_names, depth_names, label_names)]
                )
        print('{}({})'.format(section, len(rgb_names)))

def normalize_depth():
    src_dir = '../dataset/kitti2/depth'
    dst_dir = '../dataset/kitti2/normalized_3C_depth'
    depth_paths = glob.glob('../dataset/kitti2/depth/*/*/*/*/*/*depth_*.png')
    assert len(depth_paths) == 42520
    for path in tqdm(depth_paths):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        normalized_depth = (depth / 65535 * 255).astype('uint8')
        normalized_3C_depth = np.stack((normalized_depth,) * 3, axis=-1)
        new_path = path.replace(src_dir, dst_dir)
        if not os.path.isdir(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))
        cv2.imwrite(new_path, normalized_3C_depth)


def makeList_depth():
    pass


def visualize_class_distribution(scene='01'):
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '{:.2f}%'.format(height * 100),
                    ha='center', va='bottom')

    assert scene in ['01', '02', '06', '18', '20']
    variation = ['15-deg-left', '15-deg-right', 'clone']
    label_paths = []
    for v in variation:
        label_paths.extend(glob.glob('../dataset/kitti2/label/Scene{}/{}/*/*/Camera_0/*'.format(scene, v)))
    counts_sum = np.zeros((14,))
    class_names = open('../data/kitti2/kitti2_names.txt', 'r').readlines()
    class_names = [n.strip() for n in class_names]
    for path in tqdm(label_paths):
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label.resize(label.size)
        counts, bins = np.histogram(label, bins=np.arange(15))
        counts_sum += counts
    probability = counts_sum / sum(counts_sum)
    fig, ax = plt.subplots()
    width = 0.75
    ind = np.arange(len(class_names))
    rect = ax.bar(ind, probability, width, color='blue')
    ax.set_xticks(ind)
    ax.set_xticklabels(class_names)
    autolabel(rect)
    plt.title('Scene{} [{}]'.format(scene, ', '.join(variation)))
    plt.xlabel('Percentage')
    plt.ylabel('Class')
    plt.savefig('Scene{} [{}].png'.format(scene, ', '.join(variation)), dpi=300, box_inches='tight')


if __name__ == '__main__':
    transform()