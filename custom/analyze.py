import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_miou(log_paths):
    """
    Plot miou every 10 epochs
    @param log_paths: paths in format ['*-10.log', '*-20.log', ...]
    @type log_paths: list
    """
    log_paths.sort()
    result = {
        'mIoU': [],
        'mAcc': [],
        'allAcc': [],
    }
    for path in log_paths:
        line = open(path, 'r').readlines()[-13]
        ret = re.findall(r"0\.\d{4}", line)
        result['mIoU'].append(float(ret[0]))
        result['mAcc'].append(float(ret[1]))
        result['allAcc'].append(float(ret[2]))
    xs = np.arange(10, len(log_paths) * 10 + 1, 10)
    plt.plot(xs, result['mIoU'])
    plt.savefig('tmp.png')
    plt.show()


def print_avg_iou(log_paths):
    """
    Print iou per class over multiple test run
    @param log_paths: list of log file path
    @type log_paths: list
    """
    all_ious = []
    run_times = len(log_paths)
    for path in log_paths:
        lines = open(path, 'r').readlines()
        class_result_lines = [line for line in lines if 'Class_' in line]
        class_names = [re.search(r'name: (?P<name>\w+).', line).group('name') for line in class_result_lines]
        ious = [float(re.search(r'accuracy (?P<iou>0\.\d{4})', line).group('iou')) for line in
                class_result_lines]
        all_ious.append(ious)

    all_ious = np.asarray(all_ious)

    print(' | '.join(['{:20}'.format('CLASS')] + ['{:5}'.format(str(t+1)) for t in range(run_times)] + ['{:5}'.format('AVERAGE')]))
    for idx, class_name in enumerate(class_names):
        print(' | '.join(
            ['{:20}'.format(class_name)] + ['{:.3f}'.format(all_ious[t, idx]) for t in range(run_times)] + [
                '{:.3f}'.format(sum(all_ious[:, idx] / run_times))]))
    mious = all_ious.mean(axis=1)
    print(' | '.join(['{:20}'.format('AVERAGE')] + ['{:.3f}'.format(miou) for miou in mious] + ['{:.3f}'.format(mious.mean())]))
