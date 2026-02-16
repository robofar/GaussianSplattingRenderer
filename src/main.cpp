#include "Camera.h"
#include "Gaussian.h"
#include "Rasterizer.h"
#include "rasterizerUtils.h"
#include "Visualize.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        const std::string gs_path = "data/model.gsbin";
        const std::string cfg_path = "config/bonsai.yaml";
        const std::string output_dir = "output";

        const GaussianSet gs = GaussianSet::FromBinaryFile(gs_path);
        const Camera camera = Camera::FromCfg(cfg_path);

        Rasterizer rasterizer;
        std::vector<std::uint8_t> rendered;
        rendered = rasterizer.RenderSplats(gs, camera);
        visualize::SaveImage(rendered, camera.width(), camera.height(), output_dir);
        std::cout << "Rendered image saved to: " << (std::filesystem::path(output_dir) / "rendered.png") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
