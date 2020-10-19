from collections import OrderedDict
class Label:
    def __init__(self, className, trainId):
        self.className = className
        self.trainId = trainId

    def __repr__(self):
        return '({}, {})'.format(self.className, self.trainId)


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
