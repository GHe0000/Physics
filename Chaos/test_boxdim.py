
import numpy as np
import matplotlib.pyplot as plt
from tools import boxdim

# ==== 各种几何/分形生成函数 ====

# 圆
def circle(n=2000):
    theta = np.linspace(0, 2*np.pi, n)
    return np.column_stack([np.cos(theta), np.sin(theta)])

# 正方形边界
def square_boundary(n=500):
    t = np.linspace(0, 1, n)
    top = np.column_stack([t, np.ones_like(t)])
    bottom = np.column_stack([t, np.zeros_like(t)])
    left = np.column_stack([np.zeros_like(t), t])
    right = np.column_stack([np.ones_like(t), t])
    return np.vstack([top, bottom, left, right])

# 填满正方形
def filled_square(n=10000):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return np.column_stack([x, y])

# 康托集 (只在x轴)
def cantor_set(level=8):
    segments = [(0, 1)]
    for _ in range(level):
        new_segments = []
        for a, b in segments:
            third = (b - a) / 3
            new_segments.append((a, a+third))
            new_segments.append((b-third, b))
        segments = new_segments
    points = []
    for a, b in segments:
        x = np.linspace(a, b, 50)
        points.append(np.column_stack([x, np.zeros_like(x)]))
    return np.vstack(points)

# Sierpinski 三角形
def sierpinski_triangle(n=50000):
    verts = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]])
    p = np.random.rand(2)
    pts = []
    for _ in range(n):
        v = verts[np.random.randint(0,3)]
        p = (p + v) / 2
        pts.append(p)
    return np.array(pts)

# Sierpinski Carpet
def sierpinski_carpet(n=50000):
    p = np.random.rand(2)
    pts = []
    for _ in range(n):
        i, j = divmod(np.random.randint(0,8), 3)
        if (i,j) == (1,1):
            continue
        p = (p + np.array([i,j])) / 3
        pts.append(p)
    return np.array(pts)

# ==== 通用计算和绘制函数 ====
def analyze(points, name, theory_dim=None, m_list=None):
    if m_list is None:
        m_list = np.arange(10, 200, 10)
    counts = [boxdim.count_occupied_boxes(points, m) for m in m_list]
    k, b = np.polyfit(np.log(m_list), np.log(counts), 1)

    # log-log 图
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(m_list, counts, color="blue", label="Data")
    ax.plot(m_list, np.exp(k*np.log(m_list)+b), "r", label=f"Fit: D≈{k:.3f}")
    ax.set_xlabel("m (divisions)")
    ax.set_ylabel("Occupied boxes")
    if theory_dim:
        ax.set_title(f"{name} (Theory D={theory_dim:.3f})")
    else:
        ax.set_title(f"{name}")
    ax.legend()
    plt.show()

    # 可视化覆盖
    fig, ax = plt.subplots()
    boxdim.visualize_boxes(points, m=30, ax=ax, facecolor="skyblue", alpha=0.4, edgecolor="gray")
    ax.plot(points[:,0], points[:,1], "k.", markersize=1)
    ax.set_aspect("equal")
    ax.set_title(f"{name} with boxes (m=30)")
    plt.show()

    return k


# ==== 测试 ====
analyze(circle(), "Circle", theory_dim=1.0)
analyze(square_boundary(), "Square boundary", theory_dim=1.0)
analyze(filled_square(), "Filled square", theory_dim=2.0)
analyze(cantor_set(), "Cantor set", theory_dim=np.log(2)/np.log(3))
analyze(sierpinski_triangle(), "Sierpinski triangle", theory_dim=np.log(3)/np.log(2))
analyze(sierpinski_carpet(), "Sierpinski carpet", theory_dim=np.log(8)/np.log(3))
