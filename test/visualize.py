import os
import  glob
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import numpy as np

dataset_path = '../dataset/cityscapes'


class Point:
    def __init__(self, depth_image, x, y):
        self.x = x
        self.y = y
        self.depth_image = depth_image
        self.pl = []

    def add_surrounding_point(self, point):
        self.pl.append(point)

    def compute(self):
        depths = [self.depth_image[p.y, p.x] for p in self.pl]
        return depths/len(depths)


def fill(depth):
    empty_list = []
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            if depth[y, x] == 0:
                empty_list.append(Point(depth, x, y))
    for y in range(h):
        for x in range(w):
            if depth[y, x] == 0:
                empty_list.append(Point(depth, x, y))


def visualize_disparity():
    def DisparityToDepth(disp):
        disp[disp > 0] = (disp[disp > 0] - 1) / 256
        disp[disp > 0] = (0.209313 * 2262.52) / disp[disp > 0]
        return disp


    disparity_path = sorted(glob.glob(os.path.join(dataset_path, 'disparity', '*', '*', '*disparity.png')))
    rgb_path = sorted(glob.glob(os.path.join(dataset_path, 'leftImg8bit', '*', '*', '*leftImg8bit.png')))
    rgb = imread(rgb_path[0])
    disp = imread(disparity_path[0])
    depth = DisparityToDepth(disp.copy())

    fill(depth)

    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(depth)
    #
    # plt.show()


if __name__ == '__main__':
    visualize_disparity()