import taichi as ti
import numpy as np
from PIL import Image


PI = 3.14159265

ti.init(arch=ti.gpu)

image_width = 600
image_height = 400

canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

fov = 60
aspect_ratio = 3/2
#cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
#cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
#cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
#cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

def get_ray(u, v):
    # Camera 设置
    lookfrom = ti.Vector([5.0, 1.0, 0.0])
    lookat = ti.Vector([2.0, 0.0, -3.0])
    vup = ti.Vector([0.0, 1.0, 0.0])
    theta = fov * (PI / 180.0)
    half_height = ti.tan(theta / 2.0)
    half_width = aspect_ratio * half_height
    cam_origin = lookfrom
    ww = (lookfrom - lookat).normalized()
    uu = (vup.cross(ww)).normalized()
    vv = ww.cross(uu)
    cam_lower_left_corner = ti.Vector([-half_width, -half_height, -1.0])
    cam_lower_left_corner = cam_origin - half_width * uu - half_height * vv - ww
    cam_horizontal = 2 * half_width * uu
    cam_vertical = 2 * half_height * vv

    # Ray 计算
    origin = cam_origin
    direction = cam_lower_left_corner + u * cam_horizontal + v * cam_vertical - cam_origin
    
    color = ti.Vector([0.0, 0.0, 0.0])
    direction = direction.normalized()
    x = origin.normalized()
    y = direction.cross(x).normalized()
    z = x.cross(y)
    
    A_no = ti.Matrix.cols([x, y, z])  # coordinate transformation new -> old
    A_on = A_no.inverse()             # coordinate transformation old -> new
    
    phi = 0.0
    dphi = 0.001

    dudphi = -ti.cos(ti.acos(x.dot(direction))) / (ti.sin(ti.acos(x.dot(direction))) * origin.norm())
    accre_l = (A_on @ y.cross(ti.Vector([0, 1, 0]))).normalized()
    accre_phi1 = ti.atan2(accre_l[2] / accre_l[0], 1) % (2 * PI)
    accre_phi2 = (ti.atan2(accre_l[2] / accre_l[0], 1) + PI) % (2 * PI)
    u = 1 / origin.norm()
    print("?")
    for i in range(10000):
        phi += dphi
        phi %= 2 * PI
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
                    print("2")
                    color += ti.Vector([1/(ti.exp((r-4.9)/0.03)+1), 2/(ti.exp((r-5)/0.3)+1)-1, -(r+3)**3*(r-5)/432])
    return color

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = get_ray(u, v)
        canvas[i, j] = color

# Main

if __name__ == "__main__":
    gui = ti.GUI("Test", res=(image_width, image_height))
    canvas.fill(0)
    while gui.running:
        render()
        gui.set_image(canvas)
        print(np.max(canvas.to_numpy()))
        gui.show()
