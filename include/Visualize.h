#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace visualize {

void SaveImage(
    const std::vector<std::uint8_t>& rendered_image,
    int width,
    int height,
    const std::filesystem::path& save_dir);

}  // namespace visualize
