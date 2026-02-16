#include "rasterizerUtils.h"
#include "Utils.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rasterizer_utils {

FrustumCullingResult FrustumCulling(
    const GaussianSet& gs,
    const Camera& camera,
    const Camera::MatrixX3f& means_cam,
    const Camera::MatrixX2f& means_image,
    bool cull_image_bounds,
    bool cull_near_far_planes,
    float z_buffer_margin,
    float fov_margin_ratio) {
    if (means_cam.rows() != means_image.rows() || means_cam.rows() != gs.means().rows()) {
        throw std::invalid_argument("FrustumCulling input size mismatch");
    }

    std::vector<Eigen::Index> kept_indices;
    kept_indices.reserve(static_cast<std::size_t>(gs.means().rows()));

    for (Eigen::Index i = 0; i < means_cam.rows(); ++i) {
        const float z = means_cam(i, 2);
        const float u = means_image(i, 0);
        const float v = means_image(i, 1);

        bool keep = true;

        if (cull_near_far_planes) {
            keep = keep && (z >= (camera.near_plane() - z_buffer_margin));
            keep = keep && (z <= (camera.far_plane() + z_buffer_margin));
        } else {
            keep = keep && (z > 0.0f);
        }

        if (fov_margin_ratio > 0.0f) {
            const float x = means_cam(i, 0);
            const float y = means_cam(i, 1);
            const float margin_x = camera.fov_x() * fov_margin_ratio;
            const float margin_y = camera.fov_y() * fov_margin_ratio;
            const float tan_x = std::tan((camera.fov_x() * 0.5f) + margin_x);
            const float tan_y = std::tan((camera.fov_y() * 0.5f) + margin_y);
            keep = keep && (std::abs(x / z) <= tan_x);
            keep = keep && (std::abs(y / z) <= tan_y);
        }

        if (cull_image_bounds) {
            keep = keep && (u >= 0.0f) && (u < static_cast<float>(camera.width()));
            keep = keep && (v >= 0.0f) && (v < static_cast<float>(camera.height()));
        }

        if (keep) {
            kept_indices.push_back(i);
        }
    }

    const Eigen::Index m = static_cast<Eigen::Index>(kept_indices.size());

    GaussianSet::MatrixX3f means_out(m, 3);
    GaussianSet::MatrixX4f quats_out(m, 4);
    GaussianSet::MatrixX3f scales_out(m, 3);
    GaussianSet::VectorXf opacities_out(m);
    std::vector<GaussianSet::MatrixX3f> sh_out;
    sh_out.reserve(static_cast<std::size_t>(m));

    Camera::MatrixX3f means_cam_out(m, 3);
    Camera::MatrixX2f means_image_out(m, 2);

    for (Eigen::Index out_i = 0; out_i < m; ++out_i) {
        const Eigen::Index in_i = kept_indices[static_cast<std::size_t>(out_i)];
        means_out.row(out_i) = gs.means().row(in_i);
        quats_out.row(out_i) = gs.quats().row(in_i);
        scales_out.row(out_i) = gs.scales().row(in_i);
        opacities_out(out_i) = gs.opacities()(in_i);
        sh_out.push_back(gs.sh_coefficients()[static_cast<std::size_t>(in_i)]);

        means_cam_out.row(out_i) = means_cam.row(in_i);
        means_image_out.row(out_i) = means_image.row(in_i);
    }

    std::cout << "[culling] before: " << gs.Size() << ", after: " << static_cast<std::size_t>(m) << "\n";

    return FrustumCullingResult{GaussianSet(means_out, quats_out, scales_out, opacities_out, sh_out), means_cam_out, means_image_out};
}

SortGaussiansResult SortGaussiansByDepthOfMeans(
    const GaussianSet& gs,
    const Camera::MatrixX3f& means_cam,
    const Camera::MatrixX2f& means_image,
    bool back_to_front) {
    if (means_cam.rows() != means_image.rows() || means_cam.rows() != gs.means().rows()) {
        throw std::invalid_argument("SortGaussiansByDepthOfMeans input size mismatch");
    }

    std::vector<Eigen::Index> sorted_indices(static_cast<std::size_t>(means_cam.rows()));
    std::iota(sorted_indices.begin(), sorted_indices.end(), static_cast<Eigen::Index>(0));

    std::sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [&](Eigen::Index a, Eigen::Index b) {
            const float da = means_cam(a, 2);
            const float db = means_cam(b, 2);
            if (back_to_front) {
                return da > db;
            }
            return da < db;
        });

    const Eigen::Index n = static_cast<Eigen::Index>(sorted_indices.size());
    GaussianSet::MatrixX3f means_out(n, 3);
    GaussianSet::MatrixX4f quats_out(n, 4);
    GaussianSet::MatrixX3f scales_out(n, 3);
    GaussianSet::VectorXf opacities_out(n);
    std::vector<GaussianSet::MatrixX3f> sh_out;
    sh_out.reserve(static_cast<std::size_t>(n));

    Camera::MatrixX3f means_cam_out(n, 3);
    Camera::MatrixX2f means_image_out(n, 2);

    for (Eigen::Index out_i = 0; out_i < n; ++out_i) {
        const Eigen::Index in_i = sorted_indices[static_cast<std::size_t>(out_i)];
        means_out.row(out_i) = gs.means().row(in_i);
        quats_out.row(out_i) = gs.quats().row(in_i);
        scales_out.row(out_i) = gs.scales().row(in_i);
        opacities_out(out_i) = gs.opacities()(in_i);
        sh_out.push_back(gs.sh_coefficients()[static_cast<std::size_t>(in_i)]);
        means_cam_out.row(out_i) = means_cam.row(in_i);
        means_image_out.row(out_i) = means_image.row(in_i);
    }

    return SortGaussiansResult{
        GaussianSet(means_out, quats_out, scales_out, opacities_out, sh_out),
        means_cam_out,
        means_image_out};
}

std::vector<Eigen::Matrix2f> Approximate2DCovariances(
    const Camera::MatrixX3f& points_cam,
    const std::vector<Eigen::Matrix3f>& covariances_cam,
    const Camera& camera,
    float eps_2d) {
    if (static_cast<Eigen::Index>(covariances_cam.size()) != points_cam.rows()) {
        throw std::invalid_argument("Approximate2DCovariances input size mismatch");
    }

    std::vector<Eigen::Matrix2f> covariances_2d;
    covariances_2d.reserve(covariances_cam.size());

    for (Eigen::Index i = 0; i < points_cam.rows(); ++i) {
        const float x = points_cam(i, 0);
        const float y = points_cam(i, 1);
        const float z = points_cam(i, 2);
        const float z_squared = z * z;

        Eigen::Matrix<float, 2, 3> J = Eigen::Matrix<float, 2, 3>::Zero();
        J(0, 0) = camera.fx() / z;
        J(0, 2) = -camera.fx() * x / z_squared;
        J(1, 1) = camera.fy() / z;
        J(1, 2) = -camera.fy() * y / z_squared;

        Eigen::Matrix2f cov_2d = J * covariances_cam[static_cast<std::size_t>(i)] * J.transpose();
        if (eps_2d > 0.0f) {
            cov_2d += eps_2d * Eigen::Matrix2f::Identity();
        }
        covariances_2d.push_back(cov_2d);
    }

    return covariances_2d;
}

std::vector<Eigen::Vector4i> ComputeAxisAlignedBoundingBoxes(
    const Camera::MatrixX2f& gaussian_means_image,
    const Camera& camera,
    const std::vector<Eigen::Matrix2f>& covariances_image,
    float sigma_multiplier,
    bool use_simple_radius) {
    if (static_cast<Eigen::Index>(covariances_image.size()) != gaussian_means_image.rows()) {
        throw std::invalid_argument("ComputeAxisAlignedBoundingBoxes input size mismatch");
    }

    std::vector<Eigen::Vector4i> aabbs;
    aabbs.reserve(covariances_image.size());

    const int max_u = camera.width() - 1;
    const int max_v = camera.height() - 1;

    for (Eigen::Index i = 0; i < gaussian_means_image.rows(); ++i) {
        const float u_mean = gaussian_means_image(i, 0);
        const float v_mean = gaussian_means_image(i, 1);
        const Eigen::Matrix2f& cov = covariances_image[static_cast<std::size_t>(i)];

        float radius_u = 0.0f;
        float radius_v = 0.0f;
        if (use_simple_radius) {
            radius_u = sigma_multiplier * std::sqrt(std::max(cov(0, 0), 0.0f));
            radius_v = sigma_multiplier * std::sqrt(std::max(cov(1, 1), 0.0f));
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig_solver(cov);
            if (eig_solver.info() != Eigen::Success) {
                throw std::runtime_error("Eigen decomposition failed in ComputeAxisAlignedBoundingBoxes");
            }
            const float max_eig = std::max(eig_solver.eigenvalues()(0), eig_solver.eigenvalues()(1));
            const float radius = sigma_multiplier * std::sqrt(std::max(max_eig, 0.0f));
            radius_u = radius;
            radius_v = radius;
        }

        int u_min = static_cast<int>(std::floor(u_mean - radius_u));
        int v_min = static_cast<int>(std::floor(v_mean - radius_v));
        int u_max = static_cast<int>(std::ceil(u_mean + radius_u));
        int v_max = static_cast<int>(std::ceil(v_mean + radius_v));

        u_min = std::clamp(u_min, 0, max_u);
        v_min = std::clamp(v_min, 0, max_v);
        u_max = std::clamp(u_max, 0, max_u);
        v_max = std::clamp(v_max, 0, max_v);

        aabbs.emplace_back(u_min, v_min, u_max, v_max);
    }

    return aabbs;
}

}  // namespace rasterizer_utils
