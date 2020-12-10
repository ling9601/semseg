import os
import glob
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import numpy as np
from collections import OrderedDict
import time

dataset_path = 'dataset/cityscapes'


class Point:
    def __init__(self, cord):
        self.x, self.y = cord
        self.pl = []
        self.dl = []

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    def __key(self):
        return self.x, self.y

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.__key() == other.__key()
        return NotImplemented

    def add_surrounding_point(self, point):
        self.pl.append(point)

    def add_depth(self, value):
        self.dl.append(value)

    @property
    def depth_num(self):
        return len(self.dl)


def get_point(point_dict, cord):
    if cord in point_dict:
        p = point_dict[cord]
    else:
        p = Point(cord)
        point_dict[cord] = p
    return p


def fill(depth):
    point_dict = {}
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            if depth[y, x] == np.inf:
                p = get_point(point_dict, (x, y))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if depth[ny, nx] == np.inf:
                            sp = get_point(point_dict, (nx, ny))
                            p.add_surrounding_point(sp)
                        else:
                            p.add_depth(depth[ny, nx])
    sorted_dict = OrderedDict(sorted(point_dict.items(), key=lambda i: i[1].depth_num))
    while sorted_dict:
        (x, y), point = sorted_dict.popitem()
        max_num = point.depth_num
        try:
            value = sum(point.dl) / len(point.dl)
        except Exception:
            breakpoint()
        depth[y, x] = value
        for p in point.pl:
            if (p.x, p.y) not in sorted_dict:
                continue
            p.add_depth(value)
            # move to the front
            if p.depth_num > max_num or p.depth_num == 1:
                sorted_dict[(p.x, p.y)] = sorted_dict.pop((p.x, p.y))

    return depth


def DisparityToDepth(disp):
    disp[disp > 0] = (disp[disp > 0] - 1) / 256
    disp = (0.209313 * 2262.52) / disp
    return disp


def visualize_disparity():
    disparity_path = sorted(glob.glob(os.path.join(dataset_path, 'disparity', '*', '*', '*disparity.png')))
    rgb_path = sorted(glob.glob(os.path.join(dataset_path, 'leftImg8bit', '*', '*', '*leftImg8bit.png')))
    rgb = imread(rgb_path[0])
    disp = imread(disparity_path[0]).astype(np.float32)
    st = time.time()
    depth = DisparityToDepth(disp.copy())
    new_depth = fill(depth.copy())
    print(time.time() - st)

    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(rgb)
    axes[0][1].imshow(disp)
    axes[1][0].imshow(depth)
    axes[1][1].imshow(new_depth)
    plt.show()


def visualize_list():
    data_root = '../dataset/cityscapes'
    list_read = open(os.path.join(data_root, 'list/fine_train.txt')).readlines()
    assert list_read is not None
    line = list_read[0].strip()
    line_split = line.split(' ')
    print(line_split)
    image_name = os.path.join(data_root, line_split[0])
    label_name = os.path.join(data_root, line_split[1])
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(imread(image_name))
    axes[1].imshow(imread(label_name))
    plt.show()


def visualize():
    rgb_path = sorted(glob.glob(os.path.join(dataset_path, 'leftImg8bit', 'train', '*', '*leftImg8bit.png')))
    color_path = sorted(glob.glob(os.path.join(dataset_path, 'gtFine', 'train', '*', '*gtFine_color.png')))
    label_path = sorted(glob.glob(os.path.join(dataset_path, 'gtFine', 'train', '*', '*labelIds.png')))
    trainlabel_path = sorted(glob.glob(os.path.join(dataset_path, 'gtFine', 'train', '*', '*labelTrainIds.png')))
    rgb = rgb_path[0]
    color = color_path[0]
    label = label_path[0]
    trainlabel = trainlabel_path[0]
    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(imread(rgb))
    axes[0][1].imshow(imread(color))
    axes[1][0].imshow(imread(label))
    axes[1][1].imshow(imread(trainlabel))
    plt.show()


if __name__ == '__main__':
    visualize_disparity()
