import glob
import re
import matplotlib.pyplot as plt
import numpy as np

log_dir = 'exp/kitti2/{}/result/test-*-*.log'

log_paths = glob.glob(log_dir.format('fusepspnet50'))
log_paths.sort()

result = {
    'mIoU': [],
    'mAcc': [],
    'allAcc': [],
}

for path in log_paths:
    line = open(path, 'r').readlines()[-13]
    ret = re.findall(r"0.\d{4}", line)
    result['mIoU'].append(float(ret[0]))
    result['mAcc'].append(float(ret[1]))
    result['allAcc'].append(float(ret[2]))

xs = np.arange(10, 151, 10)
plt.plot(xs, result['mIoU'])
# plt.plot(xs, result['mAcc'])
# plt.plot(xs, result['allAcc'])
plt.show()