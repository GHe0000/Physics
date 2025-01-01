import numpy as np
from PIL import Image

image_width = 600
image_height = 400

canvas = np.zeros((image_width,image_height,3))

fov = 60
aspect_ratio = 3/2

def normalized(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0
    else:
        return v / norm

def get_ray(u, v):
    lookfrom = np.array([5.0, 1.0, 0.0])
    lookat = np.array([2.0, 0.0, -3.0])
    vup = np.array([0.0, 1.0, 0,0])
    theta = fov * (np.pi / 180.0)
    half_height = np.tan(theta / 2.0)
    half_width = aspect_ratio * half_height
    cam_origin = lookfrom
    ww = normalized(lookfrom - lookat)
    uu = normalized(np.cross(vup, ww))
    vv = np.cross(ww,uu)
    cam_lower_left_corner = np.array([-half_width, -half_height, -1.0])
    cam_lower_left_corner = cam_origin - half_width * uu - half_height * vv - ww
    cam_horizontal = 2 * half_width * uu
    cam_vertical = 2 * half_height * vv
    
    # Ray 计算
    origin = cam_origin
    direction = cam_lower_left_corner + u * cam_horizontal + v * cam_vertical - cam_origin

    color = np.array([0.0, 0.0, 0.0])
    direction = normalized(direction)
    x = normalized(x)
    y = normalized(np.cross(direction, x))
    z = x.cross(y)

    A_no = 
    A_on = np.linalg.inv(A_no)

