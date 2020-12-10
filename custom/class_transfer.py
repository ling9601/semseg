import imageio
import glob
from tqdm import tqdm
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import cv2

cityscapes_dir = 'dataset/cityscapes'
cityscapes_test_list_path = 'dataset/cityscapes/list/fine_val.txt'
cityscapes_to_kitti2_dict = {
    0: 5,
    2: 4,
    5: 8,
    6: 7,
    7: 6,
    8: 3,
    9: 0,
    10: 1,
    13: 10,
    14: 9,
}


def label_transform(label, transform_dict, ignore_label=255):
    """
    Replace element according to transorm_dict
    Args:
        label (): 2D np.array
        transform_dict (): dictionary
        ignore_label (): int

    Returns: new label with the same shape

    """
    new_label = label.ravel()
    new_label = np.array(
        [transform_dict[i] if i in transform_dict.keys() else ignore_label for i in new_label]).reshape(label.shape)
    assert label.shape == new_label.shape
    return new_label


def cityscapes_to_kitti2_label():
    """Create label file
    Use the following command to create symbolic link 'kitt2_to_cityscapes' under 'dataset/' first.
    'ln -s ~/dataset/kitti2_to_cityscapes dataset/kitti2_to_cityscapes'
    """

    out_dir = 'dataset/kitti2_to_cityscapes/label'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(cityscapes_test_list_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 500
        for line in tqdm(lines):
            label_path = line.split(' ')[1].strip()
            label = imageio.imread(os.path.join(cityscapes_dir, label_path))
            st = time.time()
            new_label = label_transform(label, cityscapes_to_kitti2_dict)
            imageio.imwrite(os.path.join(out_dir, os.path.basename(label_path)), new_label)


def DisparityToDepth(disp):
    disp[disp > 0] = (disp[disp > 0] - 1) / 256
    disp = (0.209313 * 2262.52) / disp
    return disp


class Point:
    def __init__(self, cord):
        self.x, self.y = cord
        self.pl = []
        self.dl = []

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    def __key(self):
        return self.x, self.y

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.__key() == other.__key()
        return NotImplemented

    def add_surrounding_point(self, point):
        self.pl.append(point)

    def add_depth(self, value):
        self.dl.append(value)

    @property
    def depth_num(self):
        return len(self.dl)


def get_point(point_dict, cord):
    if cord in point_dict:
        p = point_dict[cord]
    else:
        p = Point(cord)
        point_dict[cord] = p
    return p


def fill_depth(depth, grid_size=3):
    assert grid_size % 2 == 1 and grid_size >= 3
    undefined_point_dict = {}
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            if depth[y, x] == np.inf:
                p = get_point(undefined_point_dict, (x, y))
                for dy in range(-(grid_size // 2), grid_size // 2 + 1):
                    for dx in range(-(grid_size // 2), grid_size // 2 + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if depth[ny, nx] == np.inf:
                            sp = get_point(undefined_point_dict, (nx, ny))
                            p.add_surrounding_point(sp)
                        else:
                            p.add_depth(depth[ny, nx])
    undefined_points = list(undefined_point_dict.values())
    max_rank = grid_size * grid_size
    rank_dict = {rank: {} for rank in range(max_rank)}
    for p in undefined_points:
        rank_dict[p.depth_num][(p.x, p.y)] = p
    # store the total point num
    rank_dict['num'] = sum([len(rank_dict[rank]) for rank in rank_dict.keys()])
    assert rank_dict['num'] == len(undefined_points)
    while rank_dict['num']:
        for rank in reversed(range(max_rank)):
            if rank_dict[rank]:
                cord, point = rank_dict[rank].popitem()
                rank_dict['num'] -= 1
                undefined_point_dict.pop(cord)
                try:
                    value = sum(point.dl) / len(point.dl)
                except Exception:
                    breakpoint()
                depth[point.y, point.x] = value
                for sp in point.pl:
                    key = (sp.x, sp.y)
                    if key not in undefined_point_dict:
                        continue
                    # move sp from rank to rank+1
                    rank_dict[sp.depth_num +
                              1][key] = rank_dict[sp.depth_num].pop(key)
                    sp.add_depth(value)
                break
    return depth


def worker_1(depth, grid_size, return_dict):
    st = time.time()
    new_depth = fill_depth(depth, grid_size)
    print(f'grid_size({grid_size}): ', time.time() - st)
    return_dict[grid_size] = new_depth


def test_multi_grid(disp_path, grid_size_list=[3, 5, 7, 9]):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = DisparityToDepth(disp.copy())
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    st = time.time()
    for grid_size in grid_size_list:
        p = multiprocessing.Process(target=worker_1, args=(depth.copy(), grid_size, return_dict))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    print('total time: ', time.time() - st)
    fig = plt.figure()
    fig.add_subplot(2, 3, 1).title.set_text('disp')
    plt.imshow(disp)
    fig.add_subplot(2, 3, 2).title.set_text('depth')
    plt.imshow(depth)
    for idx, (grid_size, new_depth) in enumerate(return_dict.items()):
        fig.add_subplot(2, 3, idx + 3).title.set_text(str(grid_size))
        plt.imshow(new_depth)
    plt.show()
    # save image
    for grid_size, new_depth in return_dict.items():
        cv2.imwrite(f'grid_size_{grid_size}.png', new_depth)


def worker_2(disp_path, grid_size, out_dir):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = DisparityToDepth(disp.copy())
    new_depth = fill_depth(depth, grid_size)
    out_path = os.path.join(out_dir, os.path.basename(disp_path))
    cv2.imwrite(out_path, new_depth)


def fill_depth_multiprocessing(grid_size=5, pronum=11):
    out_dir = 'dataset/kitti2_to_cityscapes/depth'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(cityscapes_test_list_path, 'r') as f:
        lines = f.readlines()
        disp_paths = [os.path.join(cityscapes_dir, line.split(' ')[0].replace('leftImg8bit', 'disparity')) for line in
                      lines]
    pbar = tqdm(total=len(disp_paths))

    def update(*args):
        pbar.update()

    with multiprocessing.Pool(pronum) as pool:
        for disp_path in disp_paths:
            pool.apply_async(worker_2, args=(disp_path, grid_size, out_dir), callback=update)
        pool.close()
        pool.join()


if __name__ == '__main__':
    fill_depth_multiprocessing(grid_size=9)
