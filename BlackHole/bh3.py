import numpy as np
import numba as nb
from PIL import Image as Image

import time

c = 1.0
G = 2e-3

n_iter =300
dt = 0.001

@nb.jit()
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

@nb.jit()
def camera_ray_trace(x,y):
    point = camera_normal+x*camera_right+y*camera_up
    ray_origin = camera_origin.copy()
    ray_position = camera_origin.copy()
    ray_direction = point - camera_origin
    ray_velocity = c*ray_direction
    ray_total_time = 0
    color = (0,0,0)
    for t in range(n_iter):
        r = bh_position - ray_position
        a = 7.0e-3*(bh_mass/np.dot(r,r))*normalize(r)
        # print(a)
        ray_prev_pos = ray_position.copy()
        ray_velocity += a*(t*dt)
        ray_velocity = c*normalize(ray_velocity)
        ray_position += ray_velocity*(t*dt) + (a/2)*(t*dt)**2
        ray_total_time += (t*dt)

        ray_bh_dist = np.linalg.norm(ray_position - bh_position)
        if 0 <= max(ray_prev_pos[1], ray_position[1]) and 0 >= min(ray_prev_pos[1], ray_position[1]):
            a = ray_prev_pos
            b = ray_position
            l = b-a
            cross_point = np.array([a[0]-(a[1]/l[1])*l[0], 0, a[2]-(a[1]/l[1])*l[2]])
            r = np.linalg.norm(cross_point - disk_origin)
            if r <= disk_outer_r and r >= disk_inner_r:
                color = time_color(ray_total_time)
                break
        elif ray_bh_dist <= bh_radius:
            break
        elif ray_bh_dist >= 15.0:
            break
    return color

# 4 ~ 19.5

@nb.jit()
def time_color(t):
    r = (510/31)*t+(-2040/31)
    b = (-510/31)*t+(9945/31)
    return (r,0,b)

def render():
    global image_pixels
    global ray_time
    ratio = float(pic_width)/pic_height
    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0/ratio, 1.0/ratio
    xstep, ystep = (x1-x0)/(pic_width-1), (y1-y0)/(pic_height-1)

    t0 = time.time()

    for j in range(pic_height):
      y = y0 + j*ystep

      if (j+1) % 10 == 0:
        print("line " + str(j+1) + "/" + str(pic_height))
        print(time.time() - t0)
        t0 = time.time()

      for i in range(pic_width):
        x = x0 + i*xstep
        image_pixels[j,i] = camera_ray_trace(x,y)

bh_position = np.array([0, 0., 0.])
bh_mass = 80
bh_radius = 2*bh_mass*G/c**2
print(bh_radius)

c_origin = np.array([0., 0.5, -9.])
c_focus = np.array([0., 0., 0.])

pic_width = 1920
pic_height = 1080

disk_origin = c_focus
disk_inner_r = 4.5*bh_radius
disk_outer_r = 15*bh_radius
disk_color = (255,255,255)

image_pixels = np.zeros((pic_height,pic_width,3))

camera_origin = c_origin.copy()
camera_direction=normalize(c_focus-c_origin)
camera_focal_length=1.2
camera_normal=camera_origin+camera_focal_length*camera_direction
camera_right=np.array([1,0,0])
camera_up=normalize(np.cross(camera_normal,camera_right))

render()

Image.fromarray(image_pixels.astype(np.uint8)).save("images/blackhole1080-t.png")
print("Done.")