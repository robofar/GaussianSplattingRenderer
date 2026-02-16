#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace utils {

using MatrixX4f = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Matrix2f = Eigen::Matrix2f;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using Vector3f = Eigen::Vector3f;

Matrix3f QuaternionToRotationMatrix(const Eigen::Vector4f& q);
MatrixX4f NormalizeQuaternions(const MatrixX4f& quats_raw);
std::pair<Matrix3f, Vector3f> RenormalizeExtrinsics(
    const Matrix3f& R_world_to_cam,
    const Vector3f& t_world_to_cam,
    const Matrix4f& normalization_transform);

MatrixX3f EvalSphericalHarmonicsColor(
    const std::vector<MatrixX3f>& sh_coefficients,
    const MatrixX3f& directions,
    int degree);

std::vector<Matrix2f> InvertBatched2x2Matrices(const std::vector<Matrix2f>& matrices);

}  // namespace utils
