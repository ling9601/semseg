import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from tqdm import tqdm
from custom.label import rgb2label, Label, color2label_komatsu600


def my_dilation(img, color, kernel_size=(5, 5), show=False):
    """


    @param img: image in rgb format
    @type img: np.ndarray
    @param kernel_size: kernel_size of cv2.dilate()
    @type kernel_size: tuple
    @param color: (R,G,B)
    @type color: tuple
    """
    my_label_dict = {
        color: Label('None', 255),
    }
    out = rgb2label(img, my_label_dict, default_label=0)
    out_dilated = cv2.dilate(out.copy(), np.ones(kernel_size, np.uint8), iterations=1)

    new_img = img.copy()
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if out_dilated[h, w] == 255:
                new_img[h, w, 0] = color[0]
                new_img[h, w, 1] = color[1]
                new_img[h, w, 2] = color[2]

    if show:
        fig = plt.figure()
        fig.add_subplot(221).title.set_text('_ori')
        plt.imshow(out)
        plt.axis('off')
        fig.add_subplot(222).title.set_text('_after')
        plt.imshow(out_dilated)
        plt.axis('off')
        fig.add_subplot(223).title.set_text('ori')
        plt.imshow(img)
        plt.axis('off')
        fig.add_subplot(224).title.set_text('after')
        plt.imshow(new_img)
        plt.axis('off')
        plt.show()
    return new_img


def transform(label_dir, seg_dir, label_dict):
    """
    transform rgb-segmentation label to train-id label
    """
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    seg_paths = glob.glob(os.path.join(seg_dir, '*'))
    for path in tqdm(seg_paths):
        label = cv2.imread(path)[:, :, ::-1]
        new_label = rgb2label(label, label_dict)
        cv2.imwrite(os.path.join(label_dir, os.path.basename(path)), new_label)


def normalize_depth(src):
    assert src.dtype == np.uint16
    normalized_src = (src / 65535 * 255).astype('uint8')
    normalized_3C_src = np.stack((normalized_src,) * 3, axis=-1)
    return normalized_3C_src


def normalize_depth_24bit(src):
    assert src.dtype == np.uint16
    depth_24bit = (src / 65535 * 16777215).astype('uint32')
    first_channel = depth_24bit // np.power(2, 16)
    second_channel = (depth_24bit - first_channel * np.power(2, 16)) // np.power(2, 8)
    third_channel = depth_24bit - first_channel * np.power(2, 16) - second_channel * np.power(2, 8)
    depth_24bit_3C = np.stack([first_channel, second_channel, third_channel], axis=-1).astype('uint8')
    return depth_24bit_3C
