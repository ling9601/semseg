import glob
import re

path = "exp/kitti2/deepfusepspnet50/result/test-20201118_153830.log"

with open(path) as f:
    lines = f.readlines()[-12:]
    for line in lines:
        iou, accuracy = re.findall(r"0\.\d+", line)
        name = re.search(r"name: \w+", line).group()[6:]
        print(f'{name} | {iou} | {accuracy}')

