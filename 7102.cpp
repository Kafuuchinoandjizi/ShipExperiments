// A C implementation of the thermal image conversion algorithm from the Python script.
// This program reads 14-bit .raw thermal images, applies an enhanced conversion
// algorithm using a guided filter, and saves the output as 8-bit .bmp files.
// It is self-contained and does not require any external libraries like OpenCV.
//
// To compile (on Windows with MSVC):
// cl thermal_image_converter.c
//
// To compile (with GCC):
// gcc thermal_image_converter.c -o thermal_image_converter -lm -fopenmp
//
// To run:
// ./thermal_image_converter.exe (or ./thermal_image_converter on Linux/macOS)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <float.h>

#include <omp.h> // 使用 omp_get_wtime()
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

// --- Data Structures for Images ---

typedef struct {
    int width;
    int height;
    float *data;
} ImageF;

typedef struct {
    int width;
    int height;
    unsigned char *data;
} ImageU8;

// --- Memory Management for Image Structs ---

ImageF* create_image_f(int width, int height) {
    // 使用 calloc 确保数据初始化为零
    ImageF *img = (ImageF*)malloc(sizeof(ImageF));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->data = (float*)calloc((size_t)width * height, sizeof(float));
    if (!img->data) {
        free(img);
        return NULL;
    }
    return img;
}

ImageU8* create_image_u8(int width, int height) {
    ImageU8 *img = (ImageU8*)malloc(sizeof(ImageU8));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->data = (unsigned char*)calloc((size_t)width * height, sizeof(unsigned char));
    if (!img->data) {
        free(img);
        return NULL;
    }
    return img;
}

void free_image_f(ImageF *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

void free_image_u8(ImageU8 *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// --- File I/O ---

long get_file_size(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

ImageF* read_raw_thermal(const char *fname) {
    double start_time = omp_get_wtime();
    long file_size = get_file_size(fname);
    if (file_size <= 0) {
        fprintf(stderr, "Error: Cannot get size of file or file is empty: %s\n", fname);
        return NULL;
    }

    FILE *f = fopen(fname, "rb");
    if (!f) {
        perror("Error opening raw file");
        return NULL;
    }

    long num_pixels = file_size / sizeof(uint16_t);
    uint16_t *raw_data = (uint16_t*)malloc(file_size);
    if (!raw_data) {
        fprintf(stderr, "Error: Memory allocation failed for raw data.\n");
        fclose(f);
        return NULL;
    }

    if (fread(raw_data, sizeof(uint16_t), num_pixels, f) != num_pixels) {
        fprintf(stderr, "Error reading raw file: %s\n", fname);
        free(raw_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    int w = 640, h = 512;
    if (num_pixels < 320 * 256) {
        w = 192; h = 160;
    } else if (num_pixels < 640 * 512) {
        w = 320; h = 256;
    }

    ImageF *img_out = create_image_f(w, h);
    if (!img_out) {
        free(raw_data);
        return NULL;
    }

    int n_pixels_in_frame = w * h;

    // 处理原始数据中的行错位问题（与Python脚本行为保持一致）
    uint16_t *temp_row = (uint16_t *)malloc(w * sizeof(uint16_t));
    if(temp_row) {
        memcpy(temp_row, raw_data + w, w * sizeof(uint16_t));
        memcpy(raw_data, temp_row, w * sizeof(uint16_t));
        free(temp_row);
    }

    // DBOut 文件的字节序处理（与Python脚本行为保持一致）
    int is_dbout = (strstr(fname, "DBOut") != NULL);

#pragma omp parallel for
    for (int i = 0; i < n_pixels_in_frame; ++i) {
        uint16_t val = raw_data[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8); // 字节序交换
        }
        img_out->data[i] = (float)(val & 0x3FFF); // 取 14-bit 数据
    }

    free(raw_data);
    double end_time = omp_get_wtime();
    printf("Read RAW file (Wall-Clock Time: %.4f s)\n", end_time - start_time);
    return img_out;
}

void write_bmp_grayscale(const char *fname, const ImageU8 *img) {
    double start_time = omp_get_wtime();
    FILE *f = fopen(fname, "wb");
    if (!f) {
        perror("Error opening bmp file for writing");
        return;
    }

    int w = img->width;
    int h = img->height;
    int row_padded = (w + 3) & (~3);
    int image_size = row_padded * h;
    int palette_size = 256 * 4;
    int file_header_size = 14;
    int info_header_size = 40;
    uint32_t off_bits = file_header_size + info_header_size + palette_size;
    uint32_t file_size = off_bits + image_size;

    unsigned char file_header[14] = {'B', 'M'};
    // Populate bfSize
    file_header[2] = (unsigned char)(file_size);
    file_header[3] = (unsigned char)(file_size >> 8);
    file_header[4] = (unsigned char)(file_size >> 16);
    file_header[5] = (unsigned char)(file_size >> 24);
    // Populate bfOffBits
    file_header[10] = (unsigned char)(off_bits);
    file_header[11] = (unsigned char)(off_bits >> 8);
    file_header[12] = (unsigned char)(off_bits >> 16);
    file_header[13] = (unsigned char)(off_bits >> 24);

    unsigned char info_header[40] = {0};
    info_header[0] = 40; // biSize
    // Populate biWidth
    info_header[4] = (unsigned char)(w);
    info_header[5] = (unsigned char)(w >> 8);
    info_header[6] = (unsigned char)(w >> 16);
    info_header[7] = (unsigned char)(w >> 24);
    // Populate biHeight
    info_header[8] = (unsigned char)(h);
    info_header[9] = (unsigned char)(h >> 8);
    info_header[10] = (unsigned char)(h >> 16);
    info_header[11] = (unsigned char)(h >> 24);
    info_header[12] = 1;  // biPlanes
    info_header[14] = 8;  // biBitCount
    // Populate biSizeImage
    info_header[20] = (unsigned char)(image_size);
    info_header[21] = (unsigned char)(image_size >> 8);
    info_header[22] = (unsigned char)(image_size >> 16);
    info_header[23] = (unsigned char)(image_size >> 24);
    // For 8-bit, biClrUsed can be 256
    info_header[32] = (unsigned char)(256);
    info_header[33] = (unsigned char)(256 >> 8);


    fwrite(file_header, 1, file_header_size, f);
    fwrite(info_header, 1, info_header_size, f);

    // --- Color Palette (Grayscale) ---
    unsigned char palette[1024];
    for(int i = 0; i < 256; i++) {
        palette[i*4 + 0] = i; // B
        palette[i*4 + 1] = i; // G
        palette[i*4 + 2] = i; // R
        palette[i*4 + 3] = 0; // Reserved
    }
    fwrite(palette, 1, palette_size, f);

    // --- Pixel Data ---
    unsigned char *row_buffer = (unsigned char*)malloc(row_padded);
    for (int i = 0; i < h; i++) {
        // BMP stores rows bottom-to-top
        memcpy(row_buffer, img->data + (h - 1 - i) * w, w);
        // 确保写入 row_padded 字节，以包含填充
        fwrite(row_buffer, 1, row_padded, f);
    }

    free(row_buffer);
    fclose(f);
    double end_time = omp_get_wtime();
    printf("Write BMP file (Wall-Clock Time: %.4f s)\n", end_time - start_time);
}

// --- Algorithm Implementation ---

// 方法1: Quick Select (来自 dsp.cpp 的优化)
// 平均 O(n)，适合大多数情况
static int partition_floats(float *arr, int left, int right, int pivotIndex) {
    float pivotValue = arr[pivotIndex];
    // Move pivot to end
    float temp = arr[pivotIndex];
    arr[pivotIndex] = arr[right];
    arr[right] = temp;

    int storeIndex = left;
    for (int i = left; i < right; i++) {
        if (arr[i] < pivotValue) {
            temp = arr[storeIndex];
            arr[storeIndex] = arr[i];
            arr[i] = temp;
            storeIndex++;
        }
    }

    // Move pivot to its final place
    temp = arr[right];
    arr[right] = arr[storeIndex];
    arr[storeIndex] = temp;

    return storeIndex;
}

static float quick_select(float *arr, int left, int right, int k) {
    if (left == right) return arr[left];

    int pivotIndex = left + (rand() % (right - left + 1));
    pivotIndex = partition_floats(arr, left, right, pivotIndex);

    if (k == pivotIndex) {
        return arr[k];
    } else if (k < pivotIndex) {
        return quick_select(arr, left, pivotIndex - 1, k);
    } else {
        return quick_select(arr, pivotIndex + 1, right, k);
    }
}

float get_percentile_quickselect(ImageF *img, float p) {
    double start_time = omp_get_wtime();
    int size = img->width * img->height;
    float *data_copy = (float*)malloc(size * sizeof(float));
    if (!data_copy) return 0.0f;

    memcpy(data_copy, img->data, size * sizeof(float));

    int index = (int)((float)(size - 1) * p * 0.01f);
    float result = quick_select(data_copy, 0, size - 1, index);

    free(data_copy);
    double end_time = omp_get_wtime();
    // 避免在循环中打印，仅在 enhanced_thermal_conversion 中统计总时间
    (void)start_time;
    (void)end_time;
    return result;
}

float get_percentile(ImageF *img, float p) {
    return get_percentile_quickselect(img, p);
}

// --- 融合操作 (Fused Operations) ---
// 减少内存访问次数，提高 cache 性能

// 融合：乘法 + 加法
// output[i] = a[i] * alpha + b[i] * beta
static inline void fused_mul_add(const float *a, const float *b, float *output,
                                 int size, float alpha, float beta) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = a[i] * alpha + b[i] * beta;
    }
}

// Implements BORDER_REFLECT_101 logic, for border handling in box filter.
static inline int border_reflect_101(int p, int len) {
    if (len == 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p - 1;
        else p = 2 * len - p - 1;
    }
    return p;
}


ImageF* box_filter(const ImageF* src, int r) {
    int w = src->width;
    int h = src->height;
    ImageF* out = create_image_f(w, h);
    if (!out) return NULL;

    ImageF* temp = create_image_f(w, h);
    if (!temp) { free_image_f(out); return NULL; }

    // Vertical pass from src to temp
    // 🚀 优化: 累加器使用 float, 减少内存带宽压力
#pragma omp parallel for
    for (int x = 0; x < w; ++x) {
        float sum = 0.0f;
        // Initial sum for the first pixel in the column
        for (int y = -r; y <= r; ++y) {
            sum += src->data[border_reflect_101(y, h) * w + x];
        }
        // 注意：temp 存储的是累积和
        temp->data[x] = sum;

        // Sliding window for the rest of the column
        for (int y = 1; y < h; ++y) {
            sum -= src->data[border_reflect_101(y - r - 1, h) * w + x];
            sum += src->data[border_reflect_101(y + r, h) * w + x];
            temp->data[y * w + x] = sum;
        }
    }

    // Horizontal pass from temp to out
    float norm = 1.0f / (float)((2 * r + 1) * (2 * r + 1));
    // 🚀 优化: 累加器使用 float, 减少内存带宽压力
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        float sum = 0.0f;
        // Initial sum for the first pixel in the row
        for (int x = -r; x <= r; ++x) {
            sum += temp->data[y * w + border_reflect_101(x, w)];
        }
        out->data[y * w] = sum * norm;

        // Sliding window for the rest of the row
        for (int x = 1; x < w; ++x) {
            sum -= temp->data[y * w + border_reflect_101(x - r - 1, w)];
            sum += temp->data[y * w + border_reflect_101(x + r, w)];
            out->data[y * w + x] = sum * norm;
        }
    }

    free_image_f(temp);
    return out;
}


ImageF* guided_filter(const ImageF* src, int radius, float eps) {
    int w = src->width;
    int h = src->height;
    int size = w * h;

    const ImageF *I = src; // Guide and source are the same

    // E[I]
    ImageF* mean_I = box_filter(I, radius);
    if (!mean_I) return NULL;

    // 1. 计算 E[I^2]
    ImageF* I_sq_temp = create_image_f(w, h); // 临时用于 I^2
    if (!I_sq_temp) { free_image_f(mean_I); return NULL; }

#pragma omp parallel for
    for(int i=0; i<size; ++i) I_sq_temp->data[i] = I->data[i] * I->data[i];

    ImageF* mean_I_sq = box_filter(I_sq_temp, radius);
    free_image_f(I_sq_temp); // 及时释放临时 I_sq_temp
    if (!mean_I_sq) { free_image_f(mean_I); return NULL; }

    // 2. 🚀 融合计算 var_I, a, 和 b。减少 OpenMP 同步和内存分配。
    // var(I) = E[I^2] - (E[I])^2
    // a = var(I) / (var(I) + eps)
    // b = (1-a) * mean_I
    ImageF* a = create_image_f(w, h);
    ImageF* b = create_image_f(w, h);
    if (!a || !b) { free_image_f(mean_I); free_image_f(mean_I_sq); free_image_f(a); free_image_f(b); return NULL; }

#pragma omp parallel for
    for(int i=0; i<size; ++i) {
        float m_I = mean_I->data[i];
        float m_I_sq = mean_I_sq->data[i];

        // 计算 var_I
        float var_I_val = m_I_sq - m_I * m_I;

        // 计算 a = var_I / (var_I + eps)
        float a_val = var_I_val / (var_I_val + eps);
        a->data[i] = a_val;

        // 计算 b = (1-a) * mean_I
        b->data[i] = (1.0f - a_val) * m_I;
    }

    // 3. 计算 E[a] 和 E[b]
    ImageF* mean_a = box_filter(a, radius);
    ImageF* mean_b = box_filter(b, radius);

    // 4. 计算 q = mean_a * I + mean_b
    ImageF* q = create_image_f(w, h);
#pragma omp parallel for
    for(int i=0; i<size; ++i) q->data[i] = mean_a->data[i] * I->data[i] + mean_b->data[i];

    // Free intermediate images
    free_image_f(mean_I);
    free_image_f(mean_I_sq);
    free_image_f(a);
    free_image_f(b);
    free_image_f(mean_a);
    free_image_f(mean_b);

    return q;
}

#ifdef __ARM_NEON
// 优化：使用 NEON 进行向量减法 (n - b)
float* __restrict__ subtract_vectors_neon(
    float* __restrict__ d, // detail_layer->data
    const float* __restrict__ n, // image_normalized->data
    const float* __restrict__ b, // base_layer->data
    int m) // size
{
    int i;
    // 1. NEON 向量化循环：处理能被4整除的部分
    #pragma omp parallel for
    for (i = 0; i <= m - 4; i += 4) {
        float32x4_t vec_n = vld1q_f32(n + i);
        float32x4_t vec_b = vld1q_f32(b + i);
        float32x4_t vec_d = vsubq_f32(vec_n, vec_b); // 并行相减：result = n - b
        vst1q_f32(d + i, vec_d); // 将结果存回内存
    }

    // 2. 标量循环：处理余下的元素 (余数处理)
    for (; i < m; ++i) {
        d[i] = n[i] - b[i];
    }

    return d;
}
#else
// 标量版本
float* __restrict__ subtract_vectors_scalar(
        float* __restrict__ d,
        const float* __restrict__ n,
        const float* __restrict__ b,
        int m)
{
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        d[i] = n[i] - b[i];
    }
    return d;
}
#define subtract_vectors_neon subtract_vectors_scalar
#endif


ImageU8* enhanced_thermal_conversion(
        ImageF* image_14bit,
        int radius,
        float eps,
        float min_percentile,
        float max_percentile,
        float base_weight,
        float detail_weight)
{
    int w = image_14bit->width;
    int h = image_14bit->height;
    int size = w * h;

    double start_time1 = omp_get_wtime();
    // 1. Normalize to [0, 1]
    ImageF *image_normalized = create_image_f(w, h);
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        image_normalized->data[i] = image_14bit->data[i] / 16383.0f; // 2^14 - 1
    }
    double end_time1 = omp_get_wtime();
    double processing_time1 = end_time1 - start_time1;
    printf("  1. Normalize to [0, 1] (Time: %.4f s)\n", processing_time1);

    double start_time2 = omp_get_wtime();
    // 2. Guided filter
    ImageF *base_layer = guided_filter(image_normalized, radius, eps);
    ImageF *detail_layer = create_image_f(w, h);

    // 使用 NEON 或标量加速减法
    detail_layer->data = subtract_vectors_neon(
            detail_layer->data,
            image_normalized->data,
            base_layer->data,
            size
    );

    double end_time2 = omp_get_wtime();
    double processing_time2 = end_time2 - start_time2;
    printf("  2. Guided filter and detail layer (Time: %.4f s)\n", processing_time2);

    double start_time3 = omp_get_wtime();
    // 3. Log map on base layer
    ImageF *base_layer_log = create_image_f(w, h);
    // 使用 M_LN2 定义以确保精度
    float c = 1.0f / (float)M_LN2;
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // 使用 logf 以获得 float 精度
        base_layer_log->data[i] = c * logf(1.0f + base_layer->data[i]);
    }
    double end_time3 = omp_get_wtime();
    double processing_time3 = end_time3 - start_time3;
    printf("  3. Log map on base layer (Time: %.4f s)\n", processing_time3);

    double start_time4 = omp_get_wtime();
    // 4. Compress and fuse
    float min_base_val = get_percentile(base_layer_log, min_percentile);
    float max_base_val = get_percentile(base_layer_log, max_percentile);

    ImageF *base_layer_compressed = create_image_f(w, h);

    // 压缩 (截断 min/max)
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float val = base_layer_log->data[i];
        if (val < min_base_val) val = min_base_val;
        if (val > max_base_val) val = max_base_val;
        base_layer_compressed->data[i] = val;
    }

    ImageF *fused_image = create_image_f(w, h);

#ifdef __ARM_NEON
    // NEON 优化融合：(权重)加权融合，是数据并行操作
    float* __restrict__ f = fused_image->data;
    float* __restrict__ blc = base_layer_compressed->data;
    float* dd = detail_layer->data;
    float32x4_t base_w_vec = vdupq_n_f32(base_weight);
    float32x4_t detail_w_vec = vdupq_n_f32(detail_weight);

    int i;
    #pragma omp parallel for
    for (i = 0; i <= size - 4; i += 4) {
        float32x4_t vec_c = vld1q_f32(blc + i);
        float32x4_t vec_d = vld1q_f32(dd + i);
        float32x4_t p1 = vmulq_f32(base_w_vec, vec_c);
        float32x4_t p2 = vmulq_f32(detail_w_vec, vec_d);
        float32x4_t fused_vec = vaddq_f32(p1, p2);
        vst1q_f32(f + i, fused_vec);
    }

    // 标量处理余数
    for (; i < size; ++i) {
        f[i] = base_weight * blc[i] + detail_weight * dd[i];
    }
#else
    // 使用融合操作 (借鉴 dsp.cpp)：一次完成乘法和加法
    // fused = base_layer_compressed * base_weight + detail_layer * detail_weight
    fused_mul_add(base_layer_compressed->data, detail_layer->data,
                  fused_image->data, size, base_weight, detail_weight);
#endif

    double end_time4 = omp_get_wtime();
    double processing_time4 = end_time4 - start_time4;
    printf("  4. Compress and fuse (Time: %.4f s)\n", processing_time4);

    double start_time5 = omp_get_wtime();
    // 5. Normalize to 8-bit

    // 优化：使用 OpenMP reduction 并行查找 min 和 max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

#pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int j = 0; j < size; ++j) {
        float val = fused_image->data[j];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    float range = max_val - min_val;
    if (range <= 1e-6f) range = 1.0f; // 避免除以零

    ImageU8 *image_8bit = create_image_u8(w, h);
    float scale = 255.0f / range;

    // 优化：向量化归一化和量化过程
    int k;
#ifdef __ARM_NEON
    float32x4_t min_v = vdupq_n_f32((float)min_val);
    float32x4_t scale_v = vdupq_n_f32((float)scale);
    float32x4_t v255 = vdupq_n_f32(255.0f);

    #pragma omp parallel for
    for (k = 0; k <= size - 8; k += 8) {
        // 加载 8 个浮点数
        float32x4_t val_low = vld1q_f32(fused_image->data + k);
        float32x4_t val_high = vld1q_f32(fused_image->data + k + 4);

        // 归一化并缩放 (val - min_val) * scale
        val_low = vsubq_f32(val_low, min_v);
        val_low = vmulq_f32(val_low, scale_v);
        val_high = vsubq_f32(val_high, min_v);
        val_high = vmulq_f32(val_high, scale_v);

        // 截断到 [0, 255]
        val_low = vmaxq_f32(val_low, vdupq_n_f32(0.0f));
        val_low = vminq_f32(val_low, v255);
        val_high = vmaxq_f32(val_high, vdupq_n_f32(0.0f));
        val_high = vminq_f32(val_high, v255);

        // 浮点转整数（vcvtq_s32_f32：截断）
        int32x4_t int_low = vcvtq_s32_f32(val_low);
        int32x4_t int_high = vcvtq_s32_f32(val_high);

        // 饱和压缩 32位 到 16位
        int16x8_t int_s16 = vcombine_s16(vmovn_s32(int_low), vmovn_s32(int_high));

        // 饱和压缩 16位 到 8位无符号 (vqmovun_s16)
        uint8x8_t final_u8 = vqmovun_s16(int_s16);

        // 存储 8 个结果
        vst1_u8(image_8bit->data + k, final_u8);
    }
#else
    // 标量版本循环开始
    k = 0;
#endif

    // 标量处理余数和非 NEON 平台
    for (; k < size; ++k) {
        float val = (fused_image->data[k] - min_val) * scale;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        // 显式截断
        image_8bit->data[k] = (unsigned char)(int)val;
    }

    double end_time5 = omp_get_wtime();
    double processing_time5 = end_time5 - start_time5;
    printf("  5. Normalize to 8-bit (Time: %.4f s)\n", processing_time5);

    // Free all intermediate images
    free_image_f(image_normalized);
    free_image_f(base_layer);
    free_image_f(detail_layer);
    free_image_f(base_layer_log);
    free_image_f(base_layer_compressed);
    free_image_f(fused_image);

    return image_8bit;
}

// --- Main Execution Logic ---

#ifdef _WIN32
void process_directory(const char *root_dir, const char *output_root_dir) {
    char search_path[MAX_PATH];
    sprintf(search_path, "%s\\*.raw", root_dir);

    WIN32_FIND_DATAA find_data;
    HANDLE h_find = FindFirstFileA(search_path, &find_data);

    if (h_find == INVALID_HANDLE_VALUE) {
        printf("Warning: No .raw files found in '%s'.\n", root_dir);
        return;
    }

    // --- Adjustable Parameters ---
    int guided_filter_radius = 16;
    float guided_filter_eps = 0.01f * 0.01f;
    float base_min_percentile = 0.005f;
    float base_max_percentile = 99.4f;
    float base_layer_weight = 0.5f;
    float detail_layer_weight = 1.0f;

    do {
        char raw_path[MAX_PATH];
        sprintf(raw_path, "%s\\%s", root_dir, find_data.cFileName);

        printf("\nProcessing file: %s\n", find_data.cFileName);
        double start_time = omp_get_wtime(); // 使用墙钟时间

        ImageF *thermal_img_14bit = read_raw_thermal(raw_path);
        if (!thermal_img_14bit) continue;

        // 优化：使用 OpenMP reduction 并行查找最大值
        float max_pix = 0.0f;
        int size = thermal_img_14bit->width * thermal_img_14bit->height;
#pragma omp parallel for reduction(max:max_pix)
        for (int i = 0; i < size; ++i) {
            if (thermal_img_14bit->data[i] > max_pix) {
                max_pix = thermal_img_14bit->data[i];
            }
        }

        ImageU8 *final_image;
        if (max_pix == 0) {
            final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
            printf("Info: File %s is all black, saving blank image.\n", find_data.cFileName);
        } else {
            final_image = enhanced_thermal_conversion(
                    thermal_img_14bit,
                    guided_filter_radius,
                    guided_filter_eps,
                    base_min_percentile,
                    base_max_percentile,
                    base_layer_weight,
                    detail_layer_weight
            );
        }

        char stem[MAX_PATH];
        strcpy(stem, find_data.cFileName);
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';

        char output_path[MAX_PATH];
        sprintf(output_path, "%s\\%s.bmp", output_root_dir, stem);

        if(final_image) {
            write_bmp_grayscale(output_path, final_image);
        }

        double end_time = omp_get_wtime();
        double processing_time = end_time - start_time;

        printf("Processed: %s -> %s.bmp (TOTAL Wall-Clock Time: %.4f s)\n", find_data.cFileName, stem, processing_time);

        free_image_f(thermal_img_14bit);
        free_image_u8(final_image);

    } while (FindNextFileA(h_find, &find_data) != 0);

    FindClose(h_find);
}
#else // POSIX implementation for directory traversal
void process_directory(const char *root_dir, const char *output_root_dir) {
    DIR *d = opendir(root_dir);
    if (!d) {
        printf("Error: Input directory not found %s\n", root_dir);
        return;
    }

    // --- Adjustable Parameters ---
    int guided_filter_radius = 8;
    float guided_filter_eps = 0.01f * 0.01f;
    float base_min_percentile = 0.005f;
    float base_max_percentile = 99.4f;
    float base_layer_weight = 0.5f;
    float detail_layer_weight = 1.0f;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        const char *ext = strrchr(dir->d_name, '.');
        // 确保是 .raw 文件
        if (ext && strcmp(ext, ".raw") == 0) {
            char raw_path[1024];
            snprintf(raw_path, sizeof(raw_path), "%s/%s", root_dir, dir->d_name);

            printf("\nProcessing file: %s\n", dir->d_name);
            double start_time = omp_get_wtime(); // 使用墙钟时间

            ImageF *thermal_img_14bit = read_raw_thermal(raw_path);
            if (!thermal_img_14bit) continue;

            // 优化：使用 OpenMP reduction 并行查找最大值
            float max_pix = 0.0f;
            int size = thermal_img_14bit->width * thermal_img_14bit->height;
            #pragma omp parallel for reduction(max:max_pix)
            for (int i = 0; i < size; ++i) {
                if (thermal_img_14bit->data[i] > max_pix) {
                    max_pix = thermal_img_14bit->data[i];
                }
            }

            ImageU8 *final_image;
            if (max_pix == 0) {
                final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
                printf("Info: File %s is all black, saving blank image.\n", dir->d_name);
            } else {
                final_image = enhanced_thermal_conversion(
                        thermal_img_14bit,
                        guided_filter_radius,
                        guided_filter_eps,
                        base_min_percentile,
                        base_max_percentile,
                        base_layer_weight,
                        detail_layer_weight
                );
            }

            char stem[256];
            strncpy(stem, dir->d_name, sizeof(stem));
            char *dot = strrchr(stem, '.');
            if (dot) *dot = '\0';

            char output_path[1024];
            snprintf(output_path, sizeof(output_path), "%s/%s.bmp", output_root_dir, stem);

            if(final_image) {
                write_bmp_grayscale(output_path, final_image);
            }

            double end_time = omp_get_wtime();
            double processing_time = end_time - start_time;

            printf("Processed: %s -> %s.bmp (TOTAL Wall-Clock Time: %.4f s)\n", dir->d_name, stem, processing_time);

            free_image_f(thermal_img_14bit);
            free_image_u8(final_image);
        }
    }
    closedir(d);
}
#endif

int main() {
    // 请根据您的实际环境修改这些路径
    // 注意：默认路径可能在您的系统上不存在，请自行修改
    const char *root_dir = "/home/fmsh/preprocess_c/Code/RAW/";
    const char *output_root_dir = "/home/fmsh/preprocess_c/Code/test_results_c_optimized_7102";

#ifdef _WIN32
    CreateDirectoryA(output_root_dir, NULL);
#else
    mkdir(output_root_dir, 0777);
#endif

    struct stat st;
    if (stat(root_dir, &st) != 0) {
        printf("Error: Input directory not found: %s\n", root_dir);
        return 1;
    }

    printf("--- Starting Enhanced Thermal Image Conversion (C Version Optimized) ---\n");
    printf("--- Input Dir: %s\n", root_dir);
    printf("--- Output Dir: %s\n", output_root_dir);

    process_directory(root_dir, output_root_dir);

    printf("\n--- All processing complete ---\n");

    return 0;
}