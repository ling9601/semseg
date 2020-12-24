import os
import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
from collections import OrderedDict
import time
import cv2
from tqdm import tqdm


def visualize_class_distribution(label_paths, class_names, title=''):
    """
    Show class distribution
    @param label_paths: list of single channel label image file path
    @type label_paths: list
    @param class_names: list of class name
    @type class_names: list
    @param title: result image title, plt.title()
    @type title: str
    """

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '{:.2f}%'.format(height * 100),
                    ha='center', va='bottom')

    count_sum = np.zeros((len(class_names, )))
    class_num = len(class_names)
    for path in tqdm(label_paths):
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        counts, bins = np.histogram(label, bins=np.arange(class_num + 1))
        count_sum += counts
    prob = count_sum / sum(count_sum)
    fig, ax = plt.subplots()
    width = 0.75
    ind = np.arange(class_num)
    rect = ax.bar(class_names, prob, width, color='blue')
    ax.set_xticks(ind)
    autolabel(rect)
    plt.title(title)
    plt.ylabel('Percentage')
    plt.xlabel('Class')
    plt.show()


def img_overlay(overlay, output, alpha=0.5):
    return cv2.addWeighted(overlay, output, img2, 1 - alpha)


def visualize_comparison(rgb_paths, depth_paths, segmentation_list, overlay=False, alpha=0.5):
    """
    Show rgb image, true segmentation image, predicted segmentation image at the same time for len(rgb_paths) times

    @param rgb_paths: list of real image file paths
    @type rgb_paths: list
    @param depth_paths: list of single-channel depth image paths
    @type depth_paths: list
    @param segmentation_list: list of ('title', paths)
    @type segmentation_list:  list
    @param overlay:
    @type overlay: bool
    @param alpha: alpha value for overlay
    @type alpha: float
    """
    assert len(segmentation_list) <= 4
    for idx_path in range(len(rgb_paths)):
        rgb_img = cv2.imread(rgb_paths[idx_path])
        depth_img = cv2.imread(depth_paths[idx_path])
        fig = plt.figure()
        fig.add_subplot(231).title.set_text('rgb')
        plt.axis('off')
        plt.imshow(rgb_img[:, :, ::-1])
        fig.add_subplot(232).title.set_text('depth')
        plt.axis('off')
        plt.imshow(depth_img[:, :, ::-1])
        for idx_seg, seg in enumerate(segmentation_list):
            assert os.path.exists(seg[1][idx_path])
            seg_img = cv2.imread(seg[1][idx_path])
            if overlay:
                seg_img = cv2.addWeighted(seg_img, alpha, rgb_img, 1 - alpha, 0)
            fig.add_subplot(2, 3, 3 + idx_seg).title.set_text(seg[0])
            plt.axis('off')
            plt.imshow(seg_img[:, :, ::-1])
        fig.suptitle(os.path.basename(rgb_paths[idx_path]))
        plt.show()

