#include "Utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace utils {

Matrix3f QuaternionToRotationMatrix(const Eigen::Vector4f& q) {
    const float qw = q(0);
    const float qx = q(1);
    const float qy = q(2);
    const float qz = q(3);

    Matrix3f R;
    R(0, 0) = 1.0f - 2.0f * (qy * qy + qz * qz);
    R(0, 1) = 2.0f * (qx * qy - qz * qw);
    R(0, 2) = 2.0f * (qx * qz + qy * qw);

    R(1, 0) = 2.0f * (qx * qy + qz * qw);
    R(1, 1) = 1.0f - 2.0f * (qx * qx + qz * qz);
    R(1, 2) = 2.0f * (qy * qz - qx * qw);

    R(2, 0) = 2.0f * (qx * qz - qy * qw);
    R(2, 1) = 2.0f * (qy * qz + qx * qw);
    R(2, 2) = 1.0f - 2.0f * (qx * qx + qy * qy);

    return R;
}

MatrixX4f NormalizeQuaternions(const MatrixX4f& quats_raw) {
    MatrixX4f quats = quats_raw;

    for (Eigen::Index i = 0; i < quats.rows(); ++i) {
        const float norm = quats.row(i).norm();
        if (norm <= 0.0f) {
            throw std::invalid_argument("quaternion norm must be positive");
        }
        quats.row(i) /= norm;
    }

    return quats;
}

std::pair<Matrix3f, Vector3f> RenormalizeExtrinsics(
    const Matrix3f& R_world_to_cam,
    const Vector3f& t_world_to_cam,
    const Matrix4f& normalization_transform) {
    Matrix4f w2c_original = Matrix4f::Identity();
    w2c_original.block<3, 3>(0, 0) = R_world_to_cam;
    w2c_original.block<3, 1>(0, 3) = t_world_to_cam;

    Matrix4f c2w_original = w2c_original.inverse();
    Matrix4f c2w_normalized = normalization_transform * c2w_original;

    const float scaling = c2w_normalized.block<1, 3>(0, 0).norm();
    if (scaling <= 0.0f) {
        throw std::invalid_argument("normalization scaling must be positive");
    }
    c2w_normalized.block<3, 3>(0, 0) /= scaling;

    const Matrix4f w2c_normalized = c2w_normalized.inverse();
    const Matrix3f R_out = w2c_normalized.block<3, 3>(0, 0);
    const Vector3f t_out = w2c_normalized.block<3, 1>(0, 3);
    return {R_out, t_out};
}

MatrixX3f EvalSphericalHarmonicsColor(
    const std::vector<MatrixX3f>& sh_coefficients,
    const MatrixX3f& directions,
    int degree) {
    constexpr float N00 = 0.28209479177387814f;
    constexpr float N1m1 = 0.4886025119029199f;
    constexpr float N10 = 0.4886025119029199f;
    constexpr float N11 = 0.4886025119029199f;
    constexpr float N2m2 = 1.0925484305920792f;
    constexpr float N2m1 = -1.0925484305920792f;
    constexpr float N20 = 0.31539156525252005f;
    constexpr float N21 = -1.0925484305920792f;
    constexpr float N22 = 0.5462742152960396f;
    constexpr float N3m3 = -0.5900435899266435f;
    constexpr float N3m2 = 2.890611442640554f;
    constexpr float N3m1 = -0.4570457994644658f;
    constexpr float N30 = 0.3731763325901154f;
    constexpr float N31 = -0.4570457994644658f;
    constexpr float N32 = 1.445305721320277f;
    constexpr float N33 = -0.5900435899266435f;

    const Eigen::Index n = directions.rows();
    if (static_cast<Eigen::Index>(sh_coefficients.size()) != n) {
        throw std::invalid_argument("EvalSphericalHarmonicsColor size mismatch");
    }
    if (degree < 0 || degree > 3) {
        throw std::invalid_argument("EvalSphericalHarmonicsColor currently supports degree in [0, 3]");
    }

    const int required_coeffs = (degree + 1) * (degree + 1);
    MatrixX3f color = MatrixX3f::Zero(n, 3);

    for (Eigen::Index i = 0; i < n; ++i) {
        const MatrixX3f& sh = sh_coefficients[static_cast<std::size_t>(i)];
        if (sh.cols() != 3 || sh.rows() < required_coeffs) {
            throw std::invalid_argument("EvalSphericalHarmonicsColor invalid SH coefficient shape");
        }

        const float x = directions(i, 0);
        const float y = directions(i, 1);
        const float z = directions(i, 2);

        color.row(i) = N00 * sh.row(0);

        if (degree >= 1) {
            const float Y1m1 = N1m1 * y;
            const float Y10 = N10 * z;
            const float Y11 = N11 * x;
            color.row(i) += sh.row(1) * Y1m1;
            color.row(i) += sh.row(2) * Y10;
            color.row(i) += sh.row(3) * Y11;

            if (degree >= 2) {
                const float xx = x * x;
                const float yy = y * y;
                const float zz = z * z;
                const float xy = x * y;
                const float xz = x * z;
                const float yz = y * z;

                const float Y2m2 = N2m2 * xy;
                const float Y2m1 = N2m1 * yz;
                const float Y20 = N20 * (2.0f * zz - xx - yy);
                const float Y21 = N21 * xz;
                const float Y22 = N22 * (xx - yy);

                color.row(i) += sh.row(4) * Y2m2;
                color.row(i) += sh.row(5) * Y2m1;
                color.row(i) += sh.row(6) * Y20;
                color.row(i) += sh.row(7) * Y21;
                color.row(i) += sh.row(8) * Y22;

                if (degree >= 3) {
                    const float Y3m3 = N3m3 * y * (3.0f * xx - yy);
                    const float Y3m2 = N3m2 * xy * z;
                    const float Y3m1 = N3m1 * y * (4.0f * zz - xx - yy);
                    const float Y30 = N30 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                    const float Y31 = N31 * x * (4.0f * zz - xx - yy);
                    const float Y32 = N32 * z * (xx - yy);
                    const float Y33 = N33 * x * (xx - 3.0f * yy);

                    color.row(i) += sh.row(9) * Y3m3;
                    color.row(i) += sh.row(10) * Y3m2;
                    color.row(i) += sh.row(11) * Y3m1;
                    color.row(i) += sh.row(12) * Y30;
                    color.row(i) += sh.row(13) * Y31;
                    color.row(i) += sh.row(14) * Y32;
                    color.row(i) += sh.row(15) * Y33;
                }
            }
        }
    }

    return (color.array() + 0.5f).max(0.0f).matrix();
}

std::vector<Matrix2f> InvertBatched2x2Matrices(const std::vector<Matrix2f>& matrices) {
    std::vector<Matrix2f> inv_matrices;
    inv_matrices.reserve(matrices.size());

    std::size_t neg_det_count = 0;
    for (const Matrix2f& mat : matrices) {
        float det = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
        if (det < 0.0f) {
            ++neg_det_count;
        }

        det = std::max(det, 1e-10f);
        const float inv_det = 1.0f / det;

        Matrix2f inv;
        inv(0, 0) = mat(1, 1) * inv_det;
        inv(0, 1) = -mat(0, 1) * inv_det;
        inv(1, 0) = -mat(1, 0) * inv_det;
        inv(1, 1) = mat(0, 0) * inv_det;
        inv_matrices.push_back(inv);
    }

    if (neg_det_count > 0) {
        std::cout << "[warning] Found " << neg_det_count
                  << " 2x2 matrices with negative determinant in InvertBatched2x2Matrices.\n";
    }

    return inv_matrices;
}

}  // namespace utils
