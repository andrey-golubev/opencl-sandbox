#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "common/require.hpp"

class PFMReader {
    enum Endian { BIG, LITTLE, OTHER };

    inline Endian endian() {
        uint32_t value = 0xdeadbeef;
        unsigned char* bytes = (unsigned char*)&value;
        return (((*bytes) ^ 0xef) == 0 ? LITTLE : ((*bytes) ^ (0xde)) == 0 ? BIG : OTHER);
    }

    inline bool swap_endian(float pfm_endian) {
        REQUIRE(pfm_endian != 0.f);
        auto end = endian();
        REQUIRE(end != OTHER);
        // from ".pfm" format:
        // pfm endian scale > 0 => big endian
        // pfm endian scale < 0 => little endian
        return ((BIG == end) && (pfm_endian < 0.f)) || ((LITTLE == end) && (pfm_endian > 0.f));
    }

    struct File {
        FILE* fp = nullptr;
        File(const char* filename) { fp = fopen(filename, "rb"); }
        ~File() {
            fclose(fp);
            fp = nullptr;
        }
    };

public:
    int rows = 0;
    int cols = 0;

    template<typename T> std::unique_ptr<T[]> read(const std::string& filename) {
        std::unique_ptr<T[]> raster;
        float pfm_endian = 0.f;
        {
            File f(filename.c_str());
            REQUIRE(f.fp != nullptr);

            char str[100];
            int _ = 0;
            _ = fscanf(f.fp, "%s", str);
            REQUIRE(strcmp(str, "Pf") == 0);  // must be equal
            _ = fscanf(f.fp, "%s", str);
            cols = std::stoi(str);
            _ = fscanf(f.fp, "%s", str);
            rows = std::stoi(str);
            int length = cols * rows;
            _ = fscanf(f.fp, "%s", str);
            pfm_endian = std::stof(str);

            _ = fseek(f.fp, 0, SEEK_END);
            long lSize = ftell(f.fp);
            long pos = lSize - cols * rows * sizeof(T);
            _ = fseek(f.fp, pos, SEEK_SET);

            // input is a sequence of pixels: grouped by row, with the pixels in each row (left
            // to right) and the rows are ordered bottom to top (a.k.a. raster)
            raster = std::make_unique<T[]>(length);
            _ = fread(raster.get(), sizeof(T), length, f.fp);
        }
        auto raster_ptr = raster.get();

        int length = cols * rows;
        auto data = std::make_unique<T[]>(length);
        auto data_ptr = data.get();
        // reverse vertically according to PFM SPEC (make image top-to-bottom)
        for (int i = 0; i < rows; i++) {
            memcpy(&data_ptr[(rows - i - 1) * (cols)], &raster_ptr[(i * (cols))],
                   (cols) * sizeof(T));
        }

        if (swap_endian(pfm_endian)) {
            union {
                T f;
                unsigned char bytes[sizeof(T)];
            } source, dest;

            for (int i = 0; i < length; ++i) {
                source.f = data_ptr[i];
                for (unsigned int k = 0, s_T = sizeof(T); k < s_T; k++)
                    dest.bytes[k] = source.bytes[s_T - k - 1];
                data_ptr[i] = dest.f;
            }
        }
        return data;
    }
};
