import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

assert len(sys.argv) == 2
model_name = sys.argv[1]

log_dir = f'exp/kitti2/{model_name}/result'
assert os.path.isdir(log_dir)

log_paths = glob.glob(f'{log_dir}/test-*-*.log')
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

xs = np.arange(10, len(log_paths)*10+1, 10)
plt.plot(xs, result['mIoU'])
# plt.plot(xs, result['mAcc'])
# plt.plot(xs, result['allAcc'])
plt.savefig('tmp.png')
plt.show()
