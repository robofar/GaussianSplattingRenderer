#include "Gaussian.h"
#include "Utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char kMagic[8] = {'G', 'S', 'R', 'A', 'W', 'V', '1', '\0'};

std::size_t MulChecked(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > (static_cast<std::size_t>(-1) / b)) {
        throw std::runtime_error("size overflow while parsing binary gaussian file");
    }
    return a * b;
}

void ReadExact(std::ifstream& in, char* dst, std::size_t num_bytes) {
    in.read(dst, static_cast<std::streamsize>(num_bytes));
    if (in.gcount() != static_cast<std::streamsize>(num_bytes)) {
        throw std::runtime_error("Unexpected EOF while reading gaussian binary file");
    }
}

template <typename T>
T ReadScalar(std::ifstream& in) {
    T value;
    ReadExact(in, reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

std::vector<float> ReadFloatVector(std::ifstream& in, std::size_t count) {
    std::vector<float> out(count);
    const std::size_t bytes = MulChecked(count, sizeof(float));
    ReadExact(in, reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

GaussianSet::MatrixX3f FloatVectorToMatrixX3f(
    const std::vector<float>& data,
    std::size_t rows,
    const std::string& name) {
    if (data.size() != MulChecked(rows, static_cast<std::size_t>(3))) {
        throw std::runtime_error(name + " has invalid data size");
    }

    GaussianSet::MatrixX3f out(static_cast<Eigen::Index>(rows), 3);
    Eigen::Map<const GaussianSet::MatrixX3f> mapped(data.data(), static_cast<Eigen::Index>(rows), 3);
    out = mapped;
    return out;
}

GaussianSet::MatrixX4f FloatVectorToMatrixX4f(
    const std::vector<float>& data,
    std::size_t rows,
    const std::string& name) {
    if (data.size() != MulChecked(rows, static_cast<std::size_t>(4))) {
        throw std::runtime_error(name + " has invalid data size");
    }

    GaussianSet::MatrixX4f out(static_cast<Eigen::Index>(rows), 4);
    Eigen::Map<const GaussianSet::MatrixX4f> mapped(data.data(), static_cast<Eigen::Index>(rows), 4);
    out = mapped;
    return out;
}

GaussianSet::VectorXf FloatVectorToVectorXf(const std::vector<float>& data) {
    GaussianSet::VectorXf out(static_cast<Eigen::Index>(data.size()));
    Eigen::Map<const GaussianSet::VectorXf> mapped(data.data(), static_cast<Eigen::Index>(data.size()));
    out = mapped;
    return out;
}

}  // namespace

GaussianSet::GaussianSet(
    const MatrixX3f& means,
    const MatrixX4f& quats,
    const MatrixX3f& scales,
    const VectorXf& opacities,
    const std::vector<MatrixX3f>& sh_coefficients
)
    : means_(means),
      quats_(quats),
      scales_(scales),
      opacities_(opacities),
      sh_coefficients_(sh_coefficients),
      sh_degree_(0) {
    if (sh_coefficients_.empty()) {
        covariances_.clear();
        return;
    }

    sh_degree_ = ComputeShDegree(static_cast<std::size_t>(sh_coefficients_[0].rows()));
    covariances_ = ComputeCovariances(quats_, scales_);
}

GaussianSet GaussianSet::FromGsplatRaw(
    const MatrixX3f& means,
    const MatrixX4f& quats_raw,
    const MatrixX3f& log_scales,
    const VectorXf& opacity_logits,
    const std::vector<MatrixX3f>& sh0,
    const std::vector<MatrixX3f>& shN) {

    const MatrixX4f quats = utils::NormalizeQuaternions(quats_raw);
    const MatrixX3f scales = ExpPerElement(log_scales);
    const VectorXf opacities = SigmoidPerElement(opacity_logits);
    const std::vector<MatrixX3f> sh_coefficients = ConcatenateSh(sh0, shN);

    return GaussianSet(means, quats, scales, opacities, sh_coefficients);
}

GaussianSet GaussianSet::FromBinaryFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open binary file: " + path);
    }

    char magic[8];
    ReadExact(in, magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Invalid binary file magic in: " + path);
    }

    const uint64_t n_u64 = ReadScalar<uint64_t>(in);
    const uint64_t s0_u64 = ReadScalar<uint64_t>(in);
    const uint64_t sn_u64 = ReadScalar<uint64_t>(in);
    const std::size_t n = static_cast<std::size_t>(n_u64);
    const std::size_t s0 = static_cast<std::size_t>(s0_u64);
    const std::size_t sn = static_cast<std::size_t>(sn_u64);

    const std::vector<float> means_flat = ReadFloatVector(in, MulChecked(n, static_cast<std::size_t>(3)));
    const std::vector<float> quats_flat = ReadFloatVector(in, MulChecked(n, static_cast<std::size_t>(4)));
    const std::vector<float> scales_flat = ReadFloatVector(in, MulChecked(n, static_cast<std::size_t>(3)));
    const std::vector<float> opacities_flat = ReadFloatVector(in, n);
    const std::vector<float> sh0_flat = ReadFloatVector(in, MulChecked(MulChecked(n, s0), static_cast<std::size_t>(3)));
    const std::vector<float> shN_flat = ReadFloatVector(in, MulChecked(MulChecked(n, sn), static_cast<std::size_t>(3)));

    // Ensure no trailing bytes remain, so the format stays strict and predictable.
    const char extra_byte = static_cast<char>(in.get());
    if (in.good()) {
        (void)extra_byte;
        throw std::runtime_error("Gaussian binary file has unexpected trailing bytes: " + path);
    }

    MatrixX3f means = FloatVectorToMatrixX3f(means_flat, n, "means");
    MatrixX4f quats_raw = FloatVectorToMatrixX4f(quats_flat, n, "quats");
    MatrixX3f scales_raw = FloatVectorToMatrixX3f(scales_flat, n, "scales");
    VectorXf opacity_logits = FloatVectorToVectorXf(opacities_flat);

    std::vector<MatrixX3f> sh0;
    sh0.reserve(n);
    std::vector<MatrixX3f> shN;
    shN.reserve(n);

    const std::size_t sh0_block = MulChecked(s0, static_cast<std::size_t>(3));
    const std::size_t shN_block = MulChecked(sn, static_cast<std::size_t>(3));
    for (std::size_t i = 0; i < n; ++i) {
        const float* sh0_ptr = sh0_flat.data() + i * sh0_block;
        MatrixX3f sh0_coeffs(static_cast<Eigen::Index>(s0), 3);
        Eigen::Map<const MatrixX3f> sh0_mapped(sh0_ptr, static_cast<Eigen::Index>(s0), 3);
        sh0_coeffs = sh0_mapped;
        sh0.push_back(sh0_coeffs);

        const float* shN_ptr = shN_flat.data() + i * shN_block;
        MatrixX3f shN_coeffs(static_cast<Eigen::Index>(sn), 3);
        Eigen::Map<const MatrixX3f> shN_mapped(shN_ptr, static_cast<Eigen::Index>(sn), 3);
        shN_coeffs = shN_mapped;
        shN.push_back(shN_coeffs);
    }

    return FromGsplatRaw(means, quats_raw, scales_raw, opacity_logits, sh0, shN);
}

std::size_t GaussianSet::Size() const {
    return static_cast<std::size_t>(means_.rows());
}

int GaussianSet::ComputeShDegree(std::size_t num_coefficients) {
    const double root = std::sqrt(static_cast<double>(num_coefficients));
    const int degree = static_cast<int>(root) - 1;

    if (degree < 0 || static_cast<std::size_t>((degree + 1) * (degree + 1)) != num_coefficients) {
        throw std::invalid_argument("invalid SH coefficient count: expected (degree + 1)^2");
    }

    return degree;
}

std::vector<GaussianSet::Matrix3f> GaussianSet::ComputeCovariances(const MatrixX4f& quats, const MatrixX3f& scales) {
    std::vector<Matrix3f> covariances;
    covariances.reserve(static_cast<std::size_t>(quats.rows()));

    for (Eigen::Index i = 0; i < quats.rows(); ++i) {
        const Eigen::Vector4f q = quats.row(i);
        const Matrix3f R = utils::QuaternionToRotationMatrix(q);

        Matrix3f S = Matrix3f::Zero();
        S(0, 0) = scales(i, 0) * scales(i, 0);
        S(1, 1) = scales(i, 1) * scales(i, 1);
        S(2, 2) = scales(i, 2) * scales(i, 2);

        covariances.push_back(R * S * R.transpose());
    }

    return covariances;
}

GaussianSet::MatrixX3f GaussianSet::ExpPerElement(const MatrixX3f& values) {
    return values.array().exp().matrix();
}

GaussianSet::VectorXf GaussianSet::SigmoidPerElement(const VectorXf& values) {
    return (1.0f / (1.0f + (-values.array()).exp())).matrix();
}

std::vector<GaussianSet::MatrixX3f> GaussianSet::ConcatenateSh(
    const std::vector<MatrixX3f>& sh0,
    const std::vector<MatrixX3f>& shN) {
    if (sh0.size() != shN.size()) {
        throw std::invalid_argument("sh0 and shN must have same number of gaussians");
    }

    std::vector<MatrixX3f> out;
    out.reserve(sh0.size());

    for (std::size_t i = 0; i < sh0.size(); ++i) {
        if (sh0[i].cols() != 3 || shN[i].cols() != 3) {
            throw std::invalid_argument("each SH block must have 3 color channels");
        }

        MatrixX3f concatenated(sh0[i].rows() + shN[i].rows(), 3);
        concatenated.topRows(sh0[i].rows()) = sh0[i];
        concatenated.bottomRows(shN[i].rows()) = shN[i];
        out.push_back(concatenated);
    }

    return out;
}
