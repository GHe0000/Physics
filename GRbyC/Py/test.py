import numpy as np
import taichi as ti

PI = 3.1415926535897932384626434


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def update_euler(self):
        color = ti.Vector([0.0, 0.0, 0.0])
        origin = self.origin
        direction = self.direction.normalized()
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

        for i in range(10000):
            phi += dphi
            phi %= 2 * PI
            dudphi += - u * (1 - 3 / 2 * u) * dphi
            u += dudphi * dphi
            r = 1/u
            if r > 500:
                color += ti.Vector([1, 1, 1])
                break
            if r < 0.01:
                color += ti.Vector([0.5, 0.5, 0.5])
                break
            if (phi - accre_phi1) * (phi - dphi - accre_phi1) <= 0 or (phi - accre_phi2) * (phi - dphi - accre_phi2) <= 0:
                # add the mapping to the accretion disk
                if 2.5 < r < 5:
                    color += ti.Vector([1/(ti.exp((r-4.9)/0.03)+1), 2/(ti.exp((r-5)/0.3)+1)-1, -(r+3)**3*(r-5)/432])
        return color

if __name__ == '__main__':
    R = Ray(ti.Vector([1.6, 0, 0]), ti.Vector([0, 1, 0]))
    color = R.update_euler()
    print(color)
