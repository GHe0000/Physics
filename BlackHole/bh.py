import numpy as np
import numba as nb
from PIL import Image as Image
import time

c = 1.0
G = 2e-3

n_iter =300
dt = 0.001

def norm(v):
    return np.linalg.norm(v)

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0: 
       return v
    return v / n

def camera_ray_trace_new(x,y,fov=50,aspect_ratio=16/9):
    lookfrom = np.array([0,1,-10])
    lookat = np.array([0,0,0])
    vup = np.array([0.0,1.0,0.0])
    theta = fov * (np.pi / 180)
    half_height = np.tan(theta /2)
    half_width = aspect_ratio * half_height
    cam_origin = lookfrom
    w = normalize(lookfrom - lookat)
    u = normalize(np.cross(vup,w))
    v = np.cross(w,u)
    cam_lower_left_corner = np.array([-half_width, -half_height, -1.0])
    cam_lower_left_corner = cam_origin - half_width * u - half_height * v - w
    cam_horizontal = 2 * half_width * u
    cam_vertical = 2 * half_height * v

    color = np.array([0,0,0])
    ray_origin = cam_origin
    ray_direction = cam_lower_left_corner \
                    + x * cam_horizontal \
                    + y * cam_vertical \
                    - cam_origin
    direction = normalize(ray_direction)
    x = normalize(ray_origin)
    y = normalize(np.cross(ray_direction,x))
    z = np.cross(x,y)
    A_no = np.column_stack((x,y,z))
    A_on = np.linalg.inv(A_no)
    phi = 0.0
    dphi = 0.001

    dudphi = -np.cos(np.arccos(np.dot(x,direction))) \
             / (np.sin(np.arccos(np.dot(x,direction))) \
             * norm(ray_origin))
    accre_l = normalize(A_on @ np.cross(y,np.array([0,1,0])))
    accre_phi1 = np.arctan2(accre_l[2] / accre_l[0],1) % (2 * np.pi)
    accre_phi2 = (np.arctan2(accre_l[2] / accre_l[0],1) + np.pi) % (2 * np.pi)
    u = 1 / norm(ray_origin)
 
    for i in range(10000):
        phi += dphi
        phi %= 2 * np.pi
        dudphi += - u * (1 - 3 / 2 * u ** 2) * dphi
        u += dudphi * dphi
        r = 1/u
        if r > 500:
            break
        if r < 0.01:
            break
        if (phi - accre_phi1) * (phi - dphi - accre_phi1) <= 0 or (phi - accre_phi2) * (phi - dphi - accre_phi2) <= 0:
            # add the mapping to the accretion disk
            if 2.5 < r < 5:
                color += np.array([1/(np.exp((r-4.9)/0.03)+1), 2/(np.exp((r-5)/0.3)+1)-1, -(r+3)**3*(r-5)/432])
                print("!")
        return color

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
                color = disk_color
                break
        elif ray_bh_dist <= bh_radius:
            break
        elif ray_bh_dist >= 15.0:
            break
    return color

def render():
    global image_pixels
    ratio = float(pic_width)/pic_height
    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0/ratio, 1.0/ratio
    xstep, ystep = (x1-x0)/(pic_width-1), (y1-y0)/(pic_height-1)

    t0 = time.time()

    for j in range(pic_height):
      if (j+1) % 10 == 0:
        print("line " + str(j+1) + "/" + str(pic_height))
        print(time.time() - t0)
        t0 = time.time()
      for i in range(pic_width):
        image_pixels[j,i] = camera_ray_trace_new(i,j)

bh_position = np.array([0, 0., 0.])
c_origin = np.array([0., 1, -10])
c_focus = np.array([0., 0., 0.])

bh_mass = 80
bh_radius = 2*bh_mass*G/c**2
print(bh_radius)


pic_width = 100
pic_height = 100

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

Image.fromarray(image_pixels.astype(np.uint8)).save("t.png")

import matplotlib.pyplot as plt

plt.figure("Image") # 图像窗口名称
plt.imshow(image_pixels)
plt.axis("off")
plt.show()
print("Done.")
