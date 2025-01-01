import numpy as np
import numba as nb

import matplotlib.pyplot as plt

import time

image_width = 300
image_height = 100

samples_per_pixel = 1

canvas = np.zeros((image_height,image_width,3))

@nb.jit()
def norm(v):
    return np.linalg.norm(v)

@nb.jit()
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v/norm

def get_ray_euler(origin,direction):
    color = np.array([0.,0.,0.])
    direction = normalize(direction)
    x = normalize(origin)
    y = normalize(np.cross(direction,x))
    z = np.cross(x,y)
    A_no = np.column_stack((x,y,z))
    A_on = np.linalg.inv(A_no)
    phi = 0.0
    dphi = 0.001

    dudphi = -np.cos(np.arccos(np.dot(x,direction))) \
             / (np.sin(np.arccos(np.dot(x,direction))) \
             * norm(origin))

    accre_l = normalize(A_on @ np.cross(y,np.array([0,1,0])))
    accre_phi1 = np.arctan2((accre_l[2]/accre_l[0]),1) % (2 * np.pi)
    accre_phi2 = (np.arctan2((accre_l[2]/accre_l[0]),1) + np.pi) % (2 * np.pi)
    u = 1 / norm(origin)

    for i in range(10000):
        phi += dphi
        phi %= 2 * np.pi
        dudphi += -u * (1-3/2 * u ** 2) * dphi
        u += dudphi * dphi
        r = 1/u
        if r > 500:
            break
        if r < 0.01:
            break
        if (phi - accre_phi1) * (phi - dphi - accre_phi1) <= 0 or (phi - accre_phi2) * (phi - dphi - accre_phi2) <= 0:
            if 2.5 < r < 5:
                color = np.array([1/(np.exp((r-4.9)/0.03)+1),\
                                  2/(np.exp((r-5)/0.3)+1)-1,\
                                  -(r+3)**3*(r-5)/432])
                break
    return color

def render():
    lookfrom = np.array([0.,1.,-10.])
    lookat = np.array([0.,0.,0.])
    vup = np.array([0.,1.,0.])
    fov = 50
    aspect_ratio = 320/240
    theta = fov * (np.pi / 180)
    half_height = np.tan(theta / 2)
    half_width = aspect_ratio * half_height
    cam_origin = lookfrom
    w = normalize(lookfrom - lookat)
    u = normalize(np.cross(vup,w))
    v = np.cross(w,u)
    cam_lower_left_corner = np.array([-half_width, -half_height, -1.0])
    cam_lower_left_corner = cam_origin - half_width * u - half_height * v - w
    cam_horizontal = 2 * half_width * u
    cam_vertical = 2 * half_height * v
    
    st = time.time()
    for i in range(image_height):
        if (i+1) % 10 == 0:
            print("line " + str(i+1) + "/" + str(image_height))
            print(time.time() - st)
            st = time.time()
        for j in range(image_width):
            u = i / image_width
            v = i / image_height
            color = np.array([0.,0.,0.])
            ray_origin = cam_origin
            ray_direction = cam_lower_left_corner \
                            + u * cam_horizontal \
                            + v * cam_vertical \
                            - cam_origin
            color += get_ray_euler(ray_origin,ray_direction)
            canvas[i,j] = color

    np.save("Save.npy",canvas)
    plt.figure("Image")
    plt.imshow(canvas)
    plt.axis("off")
    plt.show()

render()
