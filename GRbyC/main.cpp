#include <cstdio>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/src/Core/Matrix.h>

typedef Eigen::Vector3d Color;

typedef struct Ray {
    Eigen::Vector3d origin;
    Eigen::Vector3d direction;
} Ray;

Color updateEuler(Ray ray) {
    Eigen::Vector3f color = Eigen::Vector3d::Zero();
    Eigen::Vector3d origin,direction;
    
    origin = ray.direction;
    direction = ray.direction.normalized();

    Eigen::Vector3d x,y,z;
    
    x = origin.normalized();
    y = direction.cross(x).normalized();
    z = x.cross(y);

    Eigen::Matrix3d Ano, Aon;
    return color;
}
