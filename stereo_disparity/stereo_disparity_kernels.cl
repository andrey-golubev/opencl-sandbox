// TODO: optimize indices in box_blur and other kernels

// helper functions:
int fix(int v, int max_v) {
    if (v < 0) {
        return -v;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}
bool out_of_bounds(int x, int l, int r) { return x < l || x > r; }

// defines (the values are replaced in runtime to ensure consitency between C++ and OpenCL)
#define EPS <re:EPS>
#define UNKNOWN_DISPARITY <re:UNKNOWN_DISPARITY>
#define MAX_WINDOW <re:MAX_WINDOW>
#define MAX_BORDER <re:MAX_BORDER>

// kernels and subroutines:
__kernel void box_blur(__global uchar* out, __global const uchar* in, int rows, int cols,
                       int k_size) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    const int center_shift = (k_size - 1) / 2;

    // prepare lines in advance
    __global const uchar* in_lines[MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        const int ii = idx_i + i - center_shift + MAX_BORDER;
        in_lines[i] = in + ii * cols;
    }

    // prepare column indices in advance
    int js[MAX_WINDOW] = {};
    for (int j = 0; j < k_size; ++j) {
        js[j] = fix(idx_j + j - center_shift, cols - 1);
    }

    // main loop
    uint sum = 0;
    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            sum += in_lines[i][js[j]];
        }
    }

    const double multiplier = 1.0 / (k_size * k_size);
    out[idx_i * cols + idx_j] = round(multiplier * sum);
}

double zncc(__global const uchar* left, uchar l_mean, __global const uchar* right, uchar r_mean,
            int rows, int cols, int k_size, int idx_i, int idx_j, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    // prepare lines in advance
    __global const uchar* left_lines[MAX_WINDOW] = {};
    __global const uchar* right_lines[MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        const int ii = idx_i + i - center_shift + MAX_BORDER;
        left_lines[i] = left + ii * cols;
        right_lines[i] = right + ii * cols;
    }

    // prepare column indices in advance
    int js1[MAX_WINDOW] = {};
    int js2[MAX_WINDOW] = {};
    for (int j = 0; j < k_size; ++j) {
        const int jj = idx_j + j - center_shift;
        js1[j] = fix(jj, cols - 1);
        js2[j] = fix(jj + d, cols - 1);
    }

    // main loop
    for (int i = 0; i < k_size; ++i) {
        // prefetch window
        __global const uchar* left_line = left_lines[i];
        prefetch(left_line + (idx_j - center_shift), k_size);
        __global const uchar* right_line = right_lines[i];
        prefetch(right_line + (idx_j - center_shift + d), k_size);

        // compute metric
        for (int j = 0; j < k_size; ++j) {
            const int left_pixel = (int)(left_line[js1[j]]) - (int)(l_mean);
            const int right_pixel = (int)(right_line[js2[j]]) - (int)(r_mean);

            sum += left_pixel * right_pixel;

            std_left += pow((double)left_pixel, 2);
            std_right += pow((double)right_pixel, 2);
        }
    }

    // ensure STD DEV >= EPS (otherwise we get Inf)
    std_left = max(sqrt(std_left), EPS);
    std_right = max(sqrt(std_right), EPS);

    const double dsum = sum;
    return dsum / (std_left * std_right);
}

__kernel void make_disparity_map(__global uchar* out, __global const uchar* left,
                                 __global const uchar* left_mean, __global const uchar* right,
                                 __global const uchar* right_mean, int rows, int cols,
                                 int window_size, int d_first, int d_last) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    int best_disparity = 0;
    // find max zncc and corresponding disparity for current pixel:
    double max_zncc = -1.0;  // zncc in range [-1, 1]

    uchar l_mean = left_mean[idx_i * cols + idx_j];
    for (int d = d_first; d <= d_last; ++d) {
        uchar r_mean = right_mean[idx_i * cols + fix(idx_j + d, cols - 1)];
        double v = zncc(left, l_mean, right, r_mean, rows, cols, window_size, idx_i, idx_j, d);
        if (max_zncc < v) {
            max_zncc = v;
            best_disparity = d;
        }
    }

    // store absolute value of disparity
    out[idx_i * cols + idx_j] = abs(best_disparity);
}

__kernel void cross_check_disparity(__global uchar* l2r, __global const uchar* r2l, int rows,
                                    int cols, int disparity) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    const int threshold = disparity / 4;

    const int l2r_pixel = l2r[idx_i * cols + idx_j];
    const int r2l_pixel = r2l[idx_i * cols + idx_j];
    if (abs(l2r_pixel - r2l_pixel) > threshold) {
        l2r[idx_i * cols + idx_j] = min(l2r_pixel, r2l_pixel);
    }
}

__kernel void fill_occlusions_disparity(__global uchar* data, int rows, int cols, int disparity) {
    const int idx_i = get_global_id(0);  // pixel index for row

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1)) {
        return;
    }

    // just pick closest non-zero value along current row

    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        const uchar pixel = data[idx_i * cols + idx_j];
        if (pixel == UNKNOWN_DISPARITY) {
            data[idx_i * cols + idx_j] = nearest_intensity;
        } else {
            nearest_intensity = pixel;
        }
    }
}
