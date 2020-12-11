import os
import glob
import matplotlib.pyplot as plt
from imageio import imread, imwrite
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


def visualize_comparison(rgb_paths, depth_paths, true_seg_paths, pred_seg_paths):
    """
    Show rgb image, true segmentation image, predicted segmentation image at the same time for len(rgb_paths) times
    @param rgb_paths: list of real image file paths
    @type rgb_paths: list
    @param depth_paths: list of single-channel depth image paths
    @type depth_paths: list
    @param true_seg_paths: list of true segmentation image paths
    @type true_seg_paths: list
    @param pred_seg_paths: list of predicted segmentation image paths
    @type pred_seg_paths: list
    """
    for rgb_path, depth_path, true_seg_path, pred_seg_path in zip(rgb_paths, depth_paths, true_seg_paths,
                                                                  pred_seg_paths):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        true_seg = cv2.imread(true_seg_path)
        pred_seg = cv2.imread(pred_seg_path)
        fig = plt.figure()
        fig.add_subplot(221).title.set_text('rgb')
        plt.imshow(rgb)
        fig.add_subplot(222).title.set_text('depth')
        plt.imshow(depth)
        fig.add_subplot(223).title.set_text('True')
        plt.imshow(true_seg)
        fig.add_subplot(224).title.set_text('Predicted')
        plt.imshow(pred_seg)
        fig.suptitle(os.path.basename(rgb_path))
        plt.show()
       