import numpy as np
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os

color2label = {
    (210, 0, 200): 0,
    (90, 200, 255): 1,
    (0, 199, 0): 2,
    (90, 240, 0): 3,
    (140, 140, 140): 4,
    (100, 60, 100): 5,
    (250, 100, 255): 6,
    (255, 255, 0): 7,
    (200, 200, 0): 8,
    (255, 130, 0): 9,
    (80, 80, 80): 10,
    (160, 60, 60): 11,
    (255, 127, 80): 12,
    (0, 139, 139): 13,
    (0, 0, 0): 255,
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
        label[img_id == c] = color2label[tuple(img[img_id == c][0])]
    return label


def transform():
    pathes = glob.glob('../dataset/kitti2/label/*/*/*/*/*/*/classgt_*.png')
    assert len(pathes) == 42520

    for path in tqdm(pathes):
        color = cv2.imread(path)[:, :, ::-1]
        label = rgb2label(color).astype('uint8')
        cv2.imwrite(path.replace('classgt', 'label'), label)


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


def makeList(section='train', variations=None, camera=0):
    """
    creat list 'train.txt', 'val.txt'
    Args:
        section: train or val
        variations: subset of [15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning, overcast, rain, sunset]
        camera: 0 or 1

    """
    if variations is None:
        variations = ['15-deg-left']
    dst = '../dataset/kitti2/list'
    rgb_names = []
    label_names = []
    for i in variations:
        rgb_names.extend(glob.glob('../dataset/kitti2/rgb/{}/*/{}/*/*/Camera_{}/rgb_*.jpg'.format(section, i, camera)))
        label_names.extend(
            glob.glob('../dataset/kitti2/label/{}/*/{}/*/*/Camera_{}/label_*.png'.format(section, i, camera)))
    assert len(rgb_names) == len(label_names)
    rgb_names = [i.replace('../dataset/kitti2/', '') for i in sorted(rgb_names)]
    label_names = [i.replace('../dataset/kitti2/', '') for i in sorted(label_names)]
    with open(os.path.join(dst, '{}.txt'.format(section)), 'w') as f:
        f.writelines([' '.join([rgb, label]) + '\n' for rgb, label in zip(rgb_names, label_names)])


if __name__ == '__main__':
    makeList()
