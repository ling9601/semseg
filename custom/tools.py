import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
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
        fig.add_subplot(222).title.set_text('_after')
        plt.imshow(out_dilated)
        fig.add_subplot(223).title.set_text('ori')
        plt.imshow(img)
        fig.add_subplot(224).title.set_text('after')
        plt.imshow(new_img)
        plt.show()
    return new_img
