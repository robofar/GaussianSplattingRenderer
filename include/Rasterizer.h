#pragma once

#include "Camera.h"
#include "Gaussian.h"
#include "rasterizerUtils.h"

#include <cstdint>
#include <vector>

class Rasterizer {
public:
    std::vector<std::uint8_t> RenderSplats(
        const GaussianSet& gs,
        const Camera& camera,
        float sigma_multiplier = 3.0f) const;
};
