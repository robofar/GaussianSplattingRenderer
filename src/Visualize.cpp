#include "Visualize.h"

#include <png.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace visualize {

void SaveImage(
    const std::vector<std::uint8_t>& rendered_image,
    int width,
    int height,
    const std::filesystem::path& save_dir) {

    std::filesystem::create_directories(save_dir);
    const std::filesystem::path file_path = save_dir / "rendered.png";

    FILE* fp = std::fopen(file_path.string().c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open output image path: " + file_path.string());
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::fclose(fp);
        throw std::runtime_error("png_create_write_struct failed");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        std::fclose(fp);
        throw std::runtime_error("libpng failed while writing: " + file_path.string());
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(
        png_ptr,
        info_ptr,
        static_cast<png_uint_32>(width),
        static_cast<png_uint_32>(height),
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    std::vector<std::uint8_t> writable_image = rendered_image;
    std::vector<png_bytep> row_ptrs(static_cast<std::size_t>(height));
    const std::size_t stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(3);
    for (int y = 0; y < height; ++y) {
        row_ptrs[static_cast<std::size_t>(y)] =
            writable_image.data() + static_cast<std::size_t>(y) * stride;
    }

    png_write_image(png_ptr, row_ptrs.data());
    png_write_end(png_ptr, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
}

}  // namespace visualize
