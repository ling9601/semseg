import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


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
    all_accs = []
    run_times = len(log_paths)
    param_lines = None
    for path in log_paths:
        lines = open(path, 'r').readlines()
        # make sure all log file have the same set of parameters
        if param_lines:
            for l1, l2 in zip(param_lines, lines[1:54]):
                assert l1 == l2
        param_lines = lines[1:54]
        class_result_lines = [line for line in lines if 'Class_' in line]
        class_names = [re.search(r'name: (?P<name>\w+).', line).group('name') for line in class_result_lines]
        ious = [float(re.search(r'accuracy (?P<iou>0\.\d{4})', line).group('iou')) for line in
                class_result_lines]
        accs = [float(re.search(r'accuracy .{6}/(?P<acc>0\.\d{4})', line).group('acc')) for line in
                class_result_lines]
        all_ious.append(ious)
        all_accs.append(accs)

    all_ious = np.asarray(all_ious)
    all_accs = np.asarray(all_accs)

    print(' | '.join(['{:20}'.format('CLASS')] + ['{:11}'.format(str(t + 1)) for t in range(run_times)] + [
        '{:11}'.format('AVERAGE')]))
    for idx, class_name in enumerate(class_names):
        print(' | '.join(
            ['{:20}'.format(class_name)] + ['{:.3f}/{:.3f}'.format(all_ious[t, idx], all_accs[t, idx]) for t in
                                            range(run_times)] + [
                '{:.3f}/{:.3f}'.format(sum(all_ious[:, idx] / run_times), sum(all_accs[:, idx] / run_times))]))
    mious = all_ious.mean(axis=1)
    maccs = all_accs.mean(axis=1)
    print(' | '.join(
        ['{:20}'.format('AVERAGE')] + ['{:.3f}/{:.3f}'.format(miou, macc) for miou, macc in zip(mious, maccs)] + [
            '{:.3f}/{:.3f}'.format(mious.mean(), maccs.mean())]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)

    args = parser.parse_args()

    log_paths = glob.glob(os.path.join(args.logdir, '*'))
    log_paths.sort()
    assert len(log_paths) > 0
    print_avg_iou(log_paths)
