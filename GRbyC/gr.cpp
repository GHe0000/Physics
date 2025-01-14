#include "gr.hpp"

Ray::Ray(Eigen::Vector3d origin, Eigen::Vector3d direction) {
    this->origin = origin;
    this->direction = direction;
}

// TODO: 这里使用的是 Euler 法求解，可以改成 Runge-Kutta 法提高精度
Color Ray::getRayColor() {
    Color color = Eigen::Vector3d::Zero();
    Eigen::Vector3d origin = this->origin;
    Eigen::Vector3d direction = this->direction.normalized();
    Eigen::Vector3d x,y,z;
    x = origin.normalized();
    y = direction.cross(x).normalized();
    z = x.cross(y);
    Eigen::Matrix3d Ano; // Ano 为从 new 到 old 的变换矩阵
    Ano << x, y, z;
    Eigen::Matrix3d Aon = Ano.inverse(); // Aon 为从 old 到 new 的变换矩阵
    
    // 迭代计算初始值设置
    double phi = 0.0;
    double dphi = 0.001;

    double dudphi = -cos(acos(x.dot(direction))) / (sin(acos(x.dot(direction))) * origin.norm());
    Eigen::Vector3d accre_l = (Aon * y.cross(Eigen::Vector3d(0,1,0))).normalized();
    //double accre_phi1 = atan2(accre_l[2], accre_l[0]) + (accre_phi1 < 0 ? 2 * PI : 0);
    //double accre_phi2 = atan2(accre_l[2], accre_l[0]) + PI + (accre_phi2 < 0 ? 2 * PI : 0);
    double accre_phi1 = fmod(atan2(accre_l[2], accre_l[0]) , 2*PI);
    double accre_phi2 = fmod(atan2(accre_l[2], accre_l[0])+PI, 2*PI);
    double u = 1 / origin.norm();
    
    // Euler 法迭代计算
    for(int i=0;i<10000;i++) {
        phi += dphi;
        phi = fmod(phi, 2*PI);
        dudphi += -u * (1 - 3.0/2.0 * u) * dphi; // 根据论文给出方程
        u += dudphi * dphi;
        double r = 1 / u;
        if (r > 500) break;
        if (r < 0.01) break;
        if (((phi - accre_phi1) * (phi - dphi - accre_phi1) <= 0) || ((phi - accre_phi2) * (phi - dphi - accre_phi2) <= 0)) { // 如果穿越吸积盘
            if (2.5 < r && r < 5) {
                // TODO: 这里应该是计算吸积盘颜色，应单独调用吸积盘函数
                color += Eigen::Vector3d(1.0 / (exp((r-4.9) / 0.03) + 1), 2.0 / (exp((r-5) / 0.3) + 1) -1, -pow(r+3,3) * (r-5)/432);
            }
        }
    }
    return color;
}

void Camera::setupCamera() {
    // TODO: 实现摄像机参数设置
}

Ray Camera::genRay(int u, int v) {
    return Ray(this->camOrigin,
            this->camLowerLeft
            + u * this->camHorizontal
            + v * this->camVertical
            - this->camOrigin);
}
