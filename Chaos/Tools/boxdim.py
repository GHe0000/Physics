'''
计盒维度相关的函数
'''

import numpy as np
import matplotlib as mpl

def get_occupied_boxes(points: np.ndarray, m: int) -> np.ndarray:
    # 归一化，但是不改变形状
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    scale = (maxs - mins).max()
    norm_points = (points - mins) / scale

    # 计算每个点所在的盒子
    ids = np.floor(norm_points * m).astype(int)
    ids = np.clip(ids, 0, m-1)

    # 去除重复
    ids = np.unique(ids, axis=0)
    return ids

def count_occupied_boxes(points: np.ndarray, m: int) -> int:
    boxes_ids = get_occupied_boxes(points, m) 
    return len(boxes_ids)

def box_counting_dimension(points: np.ndarray, m_list: np.ndarray) -> float:
    counts = [count_occupied_boxes(points, m) for m in m_list]
    k, b = np.polyfit(np.log(m_list), np.log(counts), 1)
    return k

def get_boxes_coordinates(points: np.ndarray, m: int):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    scale = (maxs - mins).max()
    ids = get_occupied_boxes(points, m)
    box_size = scale / m
    coords = mins + ids * box_size
    return coords, box_size

def visualize_boxes(points: np.ndarray, m: int, ax, **kwargs):
    covered_boxes, box_size = get_boxes_coordinates(points, m)
    patches = [mpl.patches.Rectangle((px, py), box_size, box_size)
               for px, py in covered_boxes]
    collection = mpl.collections.PatchCollection(patches, **kwargs)
    ax.add_collection(collection)

