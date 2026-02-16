#pragma once

#include "Camera.h"
#include "Gaussian.h"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rasterizer_utils {

struct FrustumCullingResult {
    GaussianSet gs;
    Camera::MatrixX3f means_cam;
    Camera::MatrixX2f means_image;
};

struct SortGaussiansResult {
    GaussianSet gs;
    Camera::MatrixX3f means_cam;
    Camera::MatrixX2f means_image;
};

FrustumCullingResult FrustumCulling(
    const GaussianSet& gs,
    const Camera& camera,
    const Camera::MatrixX3f& means_cam,
    const Camera::MatrixX2f& means_image,
    bool cull_image_bounds = false,
    bool cull_near_far_planes = false,
    float z_buffer_margin = 0.0f,
    float fov_margin_ratio = 0.0f);

SortGaussiansResult SortGaussiansByDepthOfMeans(
    const GaussianSet& gs,
    const Camera::MatrixX3f& means_cam,
    const Camera::MatrixX2f& means_image,
    bool back_to_front = true);

std::vector<Eigen::Matrix2f> Approximate2DCovariances(
    const Camera::MatrixX3f& points_cam,
    const std::vector<Eigen::Matrix3f>& covariances_cam,
    const Camera& camera,
    float eps_2d = 0.3f);

std::vector<Eigen::Vector4i> ComputeAxisAlignedBoundingBoxes(
    const Camera::MatrixX2f& gaussian_means_image,
    const Camera& camera,
    const std::vector<Eigen::Matrix2f>& covariances_image,
    float sigma_multiplier,
    bool use_simple_radius = true);

}  // namespace rasterizer_utils
