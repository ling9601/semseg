import numpy as np


class Label:
    def __init__(self, className, trainId):
        self.className = className
        self.trainId = trainId

    def __repr__(self):
        return '({}, {})'.format(self.className, self.trainId)


def rgb2label(img, label_dict):
    """

    Args:
        label_dict:
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
        label[img_id == c] = label_dict[tuple(img[img_id == c][0])].trainId
    return label.astype('uint8')


color2label = {
    (210, 0, 200): Label('Terrain', 0),
    (90, 200, 255): Label('Sky', 1),
    (0, 199, 0): Label('Tree', 2),
    (90, 240, 0): Label('Vegetation', 3),
    (140, 140, 140): Label('Building', 4),
    (100, 60, 100): Label('Road', 5),
    (250, 100, 255): Label('GuardRail', 6),
    (255, 255, 0): Label('TrafficSign', 7),
    (200, 200, 0): Label('TrafficLight', 8),
    (255, 130, 0): Label('Pole', 9),
    (80, 80, 80): Label('Misc', 10),
    (160, 60, 60): Label('Truck', 11),
    (255, 127, 80): Label('Car', 12),
    (0, 139, 139): Label('Van', 13),
    (0, 0, 0): Label('Undefined', 255),
}

color2label_scene02 = {
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

color2label_komatsu600 = {
    (0, 0, 255): Label('Road', 0),
    (255, 255, 0): Label('RoughRoad', 1),
    (255, 0, 0): Label('Berm', 2),
    (173, 216, 230): Label('Puddle', 3),
    (105, 105, 105): Label('others', 4),
    (211, 211, 211): Label('Sky', 5),
    (255, 165, 0): Label('ConstructionVehicle', 6),
    (255, 192, 203): Label('Prop', 7),
    (221, 160, 221): Label('Building', 8),
    (34, 139, 34): Label('Foliage', 9),
    (240, 230, 140): Label('Rock', 10)
}
