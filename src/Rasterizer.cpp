#include "Rasterizer.h"
#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr std::size_t RGB_CHANNELS = 3;
constexpr std::uint8_t BLACK = 0;
}  // namespace

std::vector<std::uint8_t> Rasterizer::RenderSplats(
    const GaussianSet& gs,
    const Camera& camera,
    float sigma_multiplier) const {
    const int width = camera.width();
    const int height = camera.height();
    const std::size_t num_pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    std::vector<float> color(num_pixels * RGB_CHANNELS, 0.0f);
    std::vector<float> alpha(num_pixels, 0.0f);

    const Camera::MatrixX3f means_cam = camera.TransformPointsWorldToCam(gs.means());
    const Camera::MatrixX2f means_image = camera.ProjectPointsCamToImagePlane(means_cam);

    const rasterizer_utils::FrustumCullingResult culled = rasterizer_utils::FrustumCulling(gs, camera, means_cam, means_image, true, true, 0.0f, 0.3f);
    const rasterizer_utils::SortGaussiansResult sorted = rasterizer_utils::SortGaussiansByDepthOfMeans(culled.gs, culled.means_cam, culled.means_image, true);

    const std::vector<Eigen::Matrix3f> covariances_cam = camera.RotateCovariancesWorldToCam(sorted.gs.covariances());
    const std::vector<Eigen::Matrix2f> covariances_image = rasterizer_utils::Approximate2DCovariances(sorted.means_cam, covariances_cam, camera, 0.3f);
    const std::vector<Eigen::Matrix2f> inv_covariances_image = utils::InvertBatched2x2Matrices(covariances_image);

    const Camera::MatrixX3f directions = camera.ViewDirectionCamToMeanWorld(sorted.gs.means());
    const Camera::MatrixX3f sh_colors = utils::EvalSphericalHarmonicsColor(sorted.gs.sh_coefficients(), directions, sorted.gs.sh_degree());

    const std::vector<Eigen::Vector4i> aabbs = rasterizer_utils::ComputeAxisAlignedBoundingBoxes(
        sorted.means_image,
        camera,
        covariances_image,
        sigma_multiplier,
        true);

    for (Eigen::Index i = 0; i < sorted.gs.means().rows(); ++i) {
        const float opacity_gaussian = sorted.gs.opacities()(i);
        if (opacity_gaussian < 0.01f) {
            continue;
        }

        const float u_mean = sorted.means_image(i, 0);
        const float v_mean = sorted.means_image(i, 1);
        const Eigen::Vector4i& aabb = aabbs[static_cast<std::size_t>(i)];
        const Eigen::Matrix2f& inv_cov = inv_covariances_image[static_cast<std::size_t>(i)];

        for (int u_pix = aabb(0); u_pix < aabb(2); ++u_pix) {
            for (int v_pix = aabb(1); v_pix < aabb(3); ++v_pix) {
                if (u_pix < 0 || u_pix >= width || v_pix < 0 || v_pix >= height) {
                    continue;
                }

                const float du = static_cast<float>(u_pix) + 0.5f - u_mean;
                const float dv = static_cast<float>(v_pix) + 0.5f - v_mean;
                const Eigen::Vector2f d(du, dv);
                const float dist_sq = d.transpose() * inv_cov * d;

                if (dist_sq > sigma_multiplier * sigma_multiplier) {
                    continue;
                }

                const float weight = std::exp(-0.5f * dist_sq);
                const float alpha_gaussian = opacity_gaussian * weight;
                const float one_minus_alpha_gaussian = 1.0f - alpha_gaussian;

                const std::size_t pix = static_cast<std::size_t>(v_pix) * static_cast<std::size_t>(width) +
                                        static_cast<std::size_t>(u_pix);
                const std::size_t base = pix * RGB_CHANNELS;

                color[base + 0] = (sh_colors(i, 0) * alpha_gaussian) + (color[base + 0] * one_minus_alpha_gaussian);
                color[base + 1] = (sh_colors(i, 1) * alpha_gaussian) + (color[base + 1] * one_minus_alpha_gaussian);
                color[base + 2] = (sh_colors(i, 2) * alpha_gaussian) + (color[base + 2] * one_minus_alpha_gaussian);
                alpha[pix] = alpha_gaussian + alpha[pix] * one_minus_alpha_gaussian;
            }
        }
    }

    std::vector<std::uint8_t> image(num_pixels * RGB_CHANNELS, BLACK);
    for (std::size_t pix = 0; pix < num_pixels; ++pix) {
        const std::size_t base = pix * RGB_CHANNELS;
        image[base + 0] = static_cast<std::uint8_t>(std::clamp(color[base + 0] * 255.0f, 0.0f, 255.0f));
        image[base + 1] = static_cast<std::uint8_t>(std::clamp(color[base + 1] * 255.0f, 0.0f, 255.0f));
        image[base + 2] = static_cast<std::uint8_t>(std::clamp(color[base + 2] * 255.0f, 0.0f, 255.0f));
    }

    return image;
}
