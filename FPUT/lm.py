import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def random_walk_1d(r,n):
    x = np.linspace(0,1,n)
    steps = np.random.uniform(-r,r,n)
    steps[0] = 0
    y = np.cumsum(steps)
    return x,y 

def circle(n):
    x_half = np.linspace(-1,1,n)
    y_upper = np.sqrt(1 - x_half ** 2)
    x = np.concatenate([x_half, x_half[::-1]])
    y = np.concatenate([y_upper, -y_upper[::-1]])
    return x,y

def get_covered_boxs(x,y,n):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    box_size = max_range / n

    grid_x_min = (x_min + x_max - max_range) / 2
    grid_y_min = (y_min + y_max - max_range) / 2

    x_norm = (x - grid_x_min) / max_range
    y_norm = (y - grid_y_min) / max_range

    grid_x = (x_norm * n).astype(int)
    grid_y = (y_norm * n).astype(int)
    
    grid_x = np.clip(grid_x,0,n-1)
    grid_y = np.clip(grid_y,0,n-1)
    
    unique_grid = np.unique(np.stack((grid_x, grid_y),axis=1),axis=0)

    boxs_x = grid_x_min + unique_grid[:,0] * box_size
    boxs_y = grid_y_min + unique_grid[:,1] * box_size
    return np.stack((boxs_x, boxs_y),axis=1), box_size


def visualize_boxs(covered_boxs, box_size, ax, **kwargs):
    patches = [mpl.patches.Rectangle((px, py), box_size, box_size)
               for px, py in covered_boxs]
    collection = mpl.collections.PatchCollection(patches, **kwargs)
    ax.add_collection(collection)

r = 0.1
n = 5000
x,y = random_walk_1d(r,n)
# x,y = circle(n)

box_size = np.arange(100,500,10)
count = np.array([len(get_covered_boxs(x,y,n)[0]) for n in box_size])

log_box_size = np.log(box_size)
log_count = np.log(count)
k, b = np.polyfit(log_box_size, log_count, 1)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter(box_size, count, color='blue')

fit_line = np.exp(k * np.log(box_size) + b)
ax.plot(box_size, fit_line, color='red')
print(f'k={k}, b={b}')
plt.show()

covered_boxs, box_size = get_covered_boxs(x,y,20)
fig, ax = plt.subplots()
visualize_boxs(covered_boxs, box_size, ax, color='skyblue', alpha=0.5)
ax.plot(x,y,'k')
plt.show()

#boxs_index = get_covered_boxs(x,y,20)



# fig, ax = plt.subplots()
# draw_covered_boxs(x,y,covered_boxs,20,ax)
# ax.plot(x,y,'k')
# plt.show()
