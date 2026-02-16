#pragma once

#include <Eigen/Dense>

#include <optional>
#include <string>
#include <utility>
#include <vector>

class Camera {
public:
    using MatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatrixX2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
    using Matrix3f = Eigen::Matrix3f;
    using Matrix4f = Eigen::Matrix4f;
    using Vector3f = Eigen::Vector3f;
    using Vector4f = Eigen::Vector4f;

    Camera(
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        const Vector4f& q_world_to_cam,
        const Vector3f& t_world_to_cam,
        int scale = 1,
        std::optional<Matrix4f> normalization_transform = std::nullopt,
        float near_plane = 0.0f,
        float far_plane = std::numeric_limits<float>::infinity());

    static Camera FromCfg(const std::string& yaml_config_path);

    MatrixX3f TransformPointsWorldToCam(const MatrixX3f& points_world) const;
    std::vector<Matrix3f> RotateCovariancesWorldToCam(const std::vector<Matrix3f>& covariances_world) const;
    MatrixX2f ProjectPointsCamToImagePlane(const MatrixX3f& points_cam) const;
    MatrixX3f ViewDirectionCamToMeanWorld(const MatrixX3f& points_world) const;

    static std::pair<float, float> ComputeFov(float fx, float fy, int width, int height);

    float fx() const { return fx_; }
    float fy() const { return fy_; }
    float cx() const { return cx_; }
    float cy() const { return cy_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int scale() const { return scale_; }
    float near_plane() const { return near_plane_; }
    float far_plane() const { return far_plane_; }
    float fov_x() const { return fov_x_; }
    float fov_y() const { return fov_y_; }

    const Vector4f& q_world_to_cam() const { return q_world_to_cam_; }
    const Vector3f& t_world_to_cam() const { return t_world_to_cam_; }
    const Matrix3f& R_world_to_cam() const { return R_world_to_cam_; }
    const Matrix3f& R_cam_to_world() const { return R_cam_to_world_; }
    const Vector3f& camera_center_world() const { return camera_center_world_; }
    const Matrix4f& T_c_w() const { return T_c_w_; }
    const Matrix4f& T_w_c() const { return T_w_c_; }

private:
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    int width_;
    int height_;
    Vector4f q_world_to_cam_;
    Vector3f t_world_to_cam_;
    int scale_;
    float near_plane_;
    float far_plane_;

    float fov_x_;
    float fov_y_;

    Matrix3f R_world_to_cam_;
    Matrix3f R_cam_to_world_;
    Vector3f camera_center_world_;
    Matrix4f T_c_w_;
    Matrix4f T_w_c_;
};
