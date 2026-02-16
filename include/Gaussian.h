#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

class GaussianSet {
public:
    using MatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatrixX4f = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
    using Matrix3f = Eigen::Matrix3f;
    using VectorXf = Eigen::VectorXf;

    GaussianSet(
        const MatrixX3f& means,
        const MatrixX4f& quats,
        const MatrixX3f& scales,
        const VectorXf& opacities,
        const std::vector<MatrixX3f>& sh_coefficients
    );

    static GaussianSet FromGsplatRaw(
        const MatrixX3f& means,
        const MatrixX4f& quats_raw,
        const MatrixX3f& log_scales,
        const VectorXf& opacity_logits,
        const std::vector<MatrixX3f>& sh0,
        const std::vector<MatrixX3f>& shN
    );

    static GaussianSet FromBinaryFile(const std::string& path);

    std::size_t Size() const;

    const MatrixX3f& means() const { return means_; }
    const MatrixX4f& quats() const { return quats_; }
    const MatrixX3f& scales() const { return scales_; }
    const VectorXf& opacities() const { return opacities_; }
    const std::vector<MatrixX3f>& sh_coefficients() const { return sh_coefficients_; }
    const std::vector<Matrix3f>& covariances() const { return covariances_; }
    int sh_degree() const { return sh_degree_; }

private:
    static std::vector<Matrix3f> ComputeCovariances(const MatrixX4f& quats, const MatrixX3f& scales);
    static int ComputeShDegree(std::size_t num_coefficients);
    static MatrixX3f ExpPerElement(const MatrixX3f& values);
    static VectorXf SigmoidPerElement(const VectorXf& values);
    static std::vector<MatrixX3f> ConcatenateSh(const std::vector<MatrixX3f>& sh0, const std::vector<MatrixX3f>& shN);

    MatrixX3f means_;
    MatrixX4f quats_;
    MatrixX3f scales_;
    VectorXf opacities_;
    std::vector<MatrixX3f> sh_coefficients_;
    int sh_degree_;
    std::vector<Matrix3f> covariances_;
};
