#include "Camera.h"
#include "Utils.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

Eigen::Vector4f ParseVector4f(const YAML::Node& node, const std::string& key) {
    Eigen::Vector4f out;
    for (int i = 0; i < 4; ++i) {
        out(i) = node[key][i].as<float>();
    }
    return out;
}

Eigen::Vector3f ParseVector3f(const YAML::Node& node, const std::string& key) {
    Eigen::Vector3f out;
    for (int i = 0; i < 3; ++i) {
        out(i) = node[key][i].as<float>();
    }
    return out;
}

std::optional<Eigen::Matrix4f> ParseMatrix4f(const YAML::Node& node, const std::string& key) {
    const YAML::Node m = node[key];

    // If key does not exist
    if (!m) {
        return std::nullopt;
    }
    Eigen::Matrix4f out;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out(r, c) = m[r][c].as<float>();
        }
    }
    return out;
}

}  // namespace

Camera::Camera(
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height,
    const Vector4f& q_world_to_cam,
    const Vector3f& t_world_to_cam,
    int scale,
    std::optional<Matrix4f> normalization_transform,
    float near_plane,
    float far_plane)
    : fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      width_(width),
      height_(height),
      q_world_to_cam_(q_world_to_cam),
      t_world_to_cam_(t_world_to_cam),
      scale_(scale),
      near_plane_(near_plane),
      far_plane_(far_plane),
      fov_x_(0.0f),
      fov_y_(0.0f) {
    if (scale_ > 1) {
        width_ = static_cast<int>(std::ceil(static_cast<float>(width_) / static_cast<float>(scale_)));
        height_ = static_cast<int>(std::ceil(static_cast<float>(height_) / static_cast<float>(scale_)));
        const float scale_x = static_cast<float>(width) / static_cast<float>(width_);
        const float scale_y = static_cast<float>(height) / static_cast<float>(height_);
        fx_ /= scale_x;
        fy_ /= scale_y;
        cx_ /= scale_x;
        cy_ /= scale_y;
    }

    const auto fov = ComputeFov(fx_, fy_, width_, height_);
    fov_x_ = fov.first;
    fov_y_ = fov.second;

    R_world_to_cam_ = utils::QuaternionToRotationMatrix(q_world_to_cam_);

    if (normalization_transform.has_value()) {
        const auto renorm = utils::RenormalizeExtrinsics(R_world_to_cam_, t_world_to_cam_, normalization_transform.value());
        R_world_to_cam_ = renorm.first;
        t_world_to_cam_ = renorm.second;
    }

    R_cam_to_world_ = R_world_to_cam_.transpose();
    camera_center_world_ = -R_world_to_cam_.transpose() * t_world_to_cam_;

    T_c_w_.setIdentity();
    T_c_w_.block<3, 3>(0, 0) = R_world_to_cam_;
    T_c_w_.block<3, 1>(0, 3) = t_world_to_cam_;

    T_w_c_ = T_c_w_.inverse();
}

Camera Camera::FromCfg(const std::string& yaml_config_path) {
    const YAML::Node root = YAML::LoadFile(yaml_config_path);
    const YAML::Node cam = root["camera"];

    return Camera(
        cam["fx"].as<float>(),
        cam["fy"].as<float>(),
        cam["cx"].as<float>(),
        cam["cy"].as<float>(),
        cam["width"].as<int>(),
        cam["height"].as<int>(),
        ParseVector4f(cam, "q_world_to_cam"),
        ParseVector3f(cam, "t_world_to_cam"),
        cam["scale"].as<int>(),
        ParseMatrix4f(cam, "normalization_transform"),
        cam["near_plane"].as<float>(),
        [](const YAML::Node& value) {
            const std::string text = value.as<std::string>();
            if (text == "inf" || text == "INF" || text == "+inf" || text == "+INF") {
                return std::numeric_limits<float>::infinity();
            }
            return value.as<float>();
        }(cam["far_plane"])
    );
}

Camera::MatrixX3f Camera::TransformPointsWorldToCam(const MatrixX3f& points_world) const {
    MatrixX3f out(points_world.rows(), 3);
    for (Eigen::Index i = 0; i < points_world.rows(); ++i) {
        Eigen::Vector4f p_h;
        p_h << points_world.row(i).transpose(), 1.0f;
        const Eigen::Vector4f p_cam_h = T_c_w_ * p_h;
        out.row(i) = p_cam_h.head<3>().transpose();
    }
    return out;
}

std::vector<Camera::Matrix3f> Camera::RotateCovariancesWorldToCam(
    const std::vector<Matrix3f>& covariances_world) const {
    std::vector<Matrix3f> out;
    out.reserve(covariances_world.size());

    for (const auto& cov : covariances_world) {
        out.push_back(R_world_to_cam_ * cov * R_world_to_cam_.transpose());
    }

    return out;
}

Camera::MatrixX2f Camera::ProjectPointsCamToImagePlane(const MatrixX3f& points_cam) const {
    MatrixX2f out(points_cam.rows(), 2);
    for (Eigen::Index i = 0; i < points_cam.rows(); ++i) {
        const float x = points_cam(i, 0);
        const float y = points_cam(i, 1);
        const float z = points_cam(i, 2);
        out(i, 0) = fx_ * (x / z) + cx_;
        out(i, 1) = fy_ * (y / z) + cy_;
    }
    return out;
}

Camera::MatrixX3f Camera::ViewDirectionCamToMeanWorld(const MatrixX3f& points_world) const {
    MatrixX3f out(points_world.rows(), 3);
    for (Eigen::Index i = 0; i < points_world.rows(); ++i) {
        const Eigen::Vector3f dir = points_world.row(i).transpose() - camera_center_world_;
        out.row(i) = (dir / dir.norm()).transpose();
    }
    return out;
}

std::pair<float, float> Camera::ComputeFov(float fx, float fy, int width, int height) {
    const float fov_x = 2.0f * std::atan(static_cast<float>(width) / (2.0f * fx));
    const float fov_y = 2.0f * std::atan(static_cast<float>(height) / (2.0f * fy));
    return {fov_x, fov_y};
}
