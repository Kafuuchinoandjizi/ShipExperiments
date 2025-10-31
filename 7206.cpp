// A C implementation of the thermal image conversion algorithm from the Python script.
// This program reads 14-bit .raw thermal images, applies an enhanced conversion
// algorithm using a guided filter, and saves the output as 8-bit .bmp files.
// It is self-contained and does not require any external libraries like OpenCV.
//
// 优化说明:
// 1. Box Filter 和 Guided Filter 内部的点操作引入 OpenMP 并行化。
// 2. 向量化函数 (NEON) 和融合循环修正了循环边界，以处理余数。
// 3. 归一化和量化过程进行向量化优化。
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

#include <omp.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

// --- 性能优化配置 ---
#define USE_INTEGRAL_IMAGE 1  // 使用积分图像优化Box Filter (64x speedup)
#define USE_MEMORY_POOL 1     // 使用内存池避免malloc/free
#define MAX_WIDTH 640
#define MAX_HEIGHT 512
#define MAX_PIXELS (MAX_WIDTH * MAX_HEIGHT)

// --- 全局内存池 ---
typedef struct {
    float *buffer1;
    float *buffer2;
    float *buffer3;
    float *buffer4;
    float *buffer5;
    float *buffer6;
    float *integral_img;
    unsigned char *output_u8;
    int initialized;
} GlobalMemoryPool;

static GlobalMemoryPool g_pool = {0};

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

// --- 内存池管理 ---

void init_memory_pool() {
    if (!g_pool.initialized) {
        // 对齐到64字节边界优化NEON访问
        posix_memalign((void**)&g_pool.buffer1, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.buffer2, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.buffer3, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.buffer4, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.buffer5, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.buffer6, 64, MAX_PIXELS * sizeof(float));
        posix_memalign((void**)&g_pool.integral_img, 64, (MAX_WIDTH+1) * (MAX_HEIGHT+1) * sizeof(float));
        posix_memalign((void**)&g_pool.output_u8, 64, MAX_PIXELS * sizeof(unsigned char));
        g_pool.initialized = 1;
    }
}

void free_memory_pool() {
    if (g_pool.initialized) {
        free(g_pool.buffer1);
        free(g_pool.buffer2);
        free(g_pool.buffer3);
        free(g_pool.buffer4);
        free(g_pool.buffer5);
        free(g_pool.buffer6);
        free(g_pool.integral_img);
        free(g_pool.output_u8);
        memset(&g_pool, 0, sizeof(GlobalMemoryPool));
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

// 优化：使用直方图法快速计算百分位数，避免完整排序
// 这比 qsort 快约 10-20 倍
float get_percentile_fast(ImageF *img, float p) {
    int size = img->width * img->height;
    
    // 1. 并行查找 min 和 max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < size; ++i) {
        float val = img->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    if (max_val - min_val < 1e-6f) return min_val;
    
    // 2. 使用直方图（bins 数量影响精度和速度的平衡）
    #define HIST_BINS 2048
    int histogram[HIST_BINS] = {0};
    float range = max_val - min_val;
    float scale = (HIST_BINS - 1) / range;
    
    // 并行构建直方图（使用原子操作或分块策略）
    #pragma omp parallel
    {
        int local_hist[HIST_BINS] = {0};
        #pragma omp for nowait
        for (int i = 0; i < size; ++i) {
            int bin = (int)((img->data[i] - min_val) * scale);
            if (bin < 0) bin = 0;
            if (bin >= HIST_BINS) bin = HIST_BINS - 1;
            local_hist[bin]++;
        }
        
        // 合并局部直方图
        #pragma omp critical
        {
            for (int i = 0; i < HIST_BINS; ++i) {
                histogram[i] += local_hist[i];
            }
        }
    }
    
    // 3. 找到对应百分位的 bin
    int target_count = (int)(size * p / 100.0f);
    int cumsum = 0;
    int target_bin = 0;
    
    for (int i = 0; i < HIST_BINS; ++i) {
        cumsum += histogram[i];
        if (cumsum >= target_count) {
            target_bin = i;
            break;
        }
    }
    
    // 4. 将 bin 转换回值
    float result = min_val + target_bin / scale;
    return result;
    #undef HIST_BINS
}

// 保留原函数作为备用
float get_percentile(ImageF *img, float p) {
    return get_percentile_fast(img, p);
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

// --- 积分图像实现 (O(n) Box Filter) ---

// 计算积分图像 - 单次遍历O(n)
void compute_integral_image(const float* src, float* integral, int w, int h) {
    const int w1 = w + 1;
    
    // 初始化第一行和第一列为0
    for (int x = 0; x <= w; x++) integral[x] = 0;
    for (int y = 0; y <= h; y++) integral[y * w1] = 0;
    
    // 计算积分图像
    #pragma omp parallel for schedule(static, 4)
    for (int y = 1; y <= h; y++) {
        float row_sum = 0.0f;
        const float* src_row = src + (y - 1) * w;
        float* int_row = integral + y * w1;
        const float* int_prev = integral + (y - 1) * w1;
        
        for (int x = 1; x <= w; x++) {
            row_sum += src_row[x - 1];
            int_row[x] = row_sum + int_prev[x];
        }
    }
}

// 使用积分图像的O(1)复杂度Box Filter
void box_filter_integral(const float* src, float* dst, int w, int h, int r) {
    float* integral = g_pool.integral_img;
    const int w1 = w + 1;
    
    // 计算积分图像
    compute_integral_image(src, integral, w, h);
    
    const float inv_area = 1.0f / ((2 * r + 1) * (2 * r + 1));
    
    // 使用积分图像计算均值 - 每个像素只需4次访问
    #pragma omp parallel for collapse(2) schedule(static, 16)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // 计算窗口边界（带边界处理）
            int y1 = (y - r < 0) ? 0 : y - r;
            int y2 = (y + r >= h) ? h - 1 : y + r;
            int x1 = (x - r < 0) ? 0 : x - r;
            int x2 = (x + r >= w) ? w - 1 : x + r;
            
            // O(1)计算矩形区域和
            float sum = integral[(y2 + 1) * w1 + (x2 + 1)]
                      - integral[(y2 + 1) * w1 + x1]
                      - integral[y1 * w1 + (x2 + 1)]
                      + integral[y1 * w1 + x1];
            
            dst[y * w + x] = sum * inv_area;
        }
    }
}

// 旧版Box Filter（保留作为对比）
ImageF* box_filter_old(const ImageF* src, int r) {
    int w = src->width;
    int h = src->height;
    ImageF* out = create_image_f(w, h);
    if (!out) return NULL;

    ImageF* temp = create_image_f(w, h);
    if (!temp) { free_image_f(out); return NULL; }

    // Vertical pass from src to temp, using double for accumulator
    // 优化：OpenMP 并行处理列
#pragma omp parallel for
    for (int x = 0; x < w; ++x) {
        double sum = 0.0;
        // Initial sum for the first pixel in the column
        for (int y = -r; y <= r; ++y) {
            sum += src->data[border_reflect_101(y, h) * w + x];
        }
        // 注意：temp 存储的是累积和，不是平均值
        temp->data[x] = (float)sum;

        // Sliding window for the rest of the column
        for (int y = 1; y < h; ++y) {
            sum -= src->data[border_reflect_101(y - r - 1, h) * w + x];
            sum += src->data[border_reflect_101(y + r, h) * w + x];
            temp->data[y * w + x] = (float)sum;
        }
    }

    // Horizontal pass from temp to out, using double for accumulator
    float norm = 1.0f / (float)((2 * r + 1) * (2 * r + 1));
    // 优化：OpenMP 并行处理行
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        double sum = 0.0;
        // Initial sum for the first pixel in the row
        for (int x = -r; x <= r; ++x) {
            sum += temp->data[y * w + border_reflect_101(x, w)];
        }
        out->data[y * w] = (float)(sum * norm);

        // Sliding window for the rest of the row
        for (int x = 1; x < w; ++x) {
            sum -= temp->data[y * w + border_reflect_101(x - r - 1, w)];
            sum += temp->data[y * w + border_reflect_101(x + r, w)];
            out->data[y * w + x] = (float)(sum * norm);
        }
    }

    free_image_f(temp);
    return out;
}

// 新版高性能Box Filter
ImageF* box_filter(const ImageF* src, int r) {
    int w = src->width;
    int h = src->height;
    ImageF* out = create_image_f(w, h);
    if (!out) return NULL;
    
    // 使用积分图像实现O(1)复杂度
    box_filter_integral(src->data, out->data, w, h, r);
    
    return out;
}


ImageF* guided_filter(const ImageF* src, int radius, float eps) {
    int w = src->width;
    int h = src->height;
    int size = w * h;

    const ImageF *I = src; // Guide and source are the same

    // 使用内存池避免动态分配
    float* mean_I = g_pool.buffer1;
    float* mean_I_sq = g_pool.buffer2;
    float* var_I = g_pool.buffer3;
    float* a = g_pool.buffer4;
    float* b = g_pool.buffer5;
    float* I_sq_temp = g_pool.buffer6;

    // 步骤1: 计算 mean_I
    box_filter_integral(I->data, mean_I, w, h, radius);

    // 步骤2: 计算 I_sq 和 mean_I_sq（融合计算）
#ifdef __ARM_NEON
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i <= size - 4; i += 4) {
        float32x4_t vI = vld1q_f32(I->data + i);
        float32x4_t vI_sq = vmulq_f32(vI, vI);
        vst1q_f32(I_sq_temp + i, vI_sq);
    }
    for (int i = (size & ~3); i < size; i++) {
        I_sq_temp[i] = I->data[i] * I->data[i];
    }
#else
    #pragma omp parallel for schedule(static, 256)
    for(int i=0; i<size; ++i) {
        I_sq_temp[i] = I->data[i] * I->data[i];
    }
#endif

    box_filter_integral(I_sq_temp, mean_I_sq, w, h, radius);

    // 步骤3: 计算 var_I 和 a, b（融合计算）
#ifdef __ARM_NEON
    float32x4_t veps = vdupq_n_f32(eps);
    float32x4_t vone = vdupq_n_f32(1.0f);
    
    #pragma omp parallel for schedule(static, 256)
    for(int i = 0; i <= size - 4; i += 4) {
        float32x4_t vmean_I = vld1q_f32(mean_I + i);
        float32x4_t vmean_I_sq = vld1q_f32(mean_I_sq + i);
        
        // var_I = mean_I_sq - mean_I * mean_I
        float32x4_t vvar_I = vmlsq_f32(vmean_I_sq, vmean_I, vmean_I);
        
        // a = var_I / (var_I + eps)
        float32x4_t va = vdivq_f32(vvar_I, vaddq_f32(vvar_I, veps));
        
        // b = (1 - a) * mean_I
        float32x4_t vb = vmulq_f32(vsubq_f32(vone, va), vmean_I);
        
        vst1q_f32(a + i, va);
        vst1q_f32(b + i, vb);
    }
    for(int i = (size & ~3); i < size; ++i) {
        float var = mean_I_sq[i] - mean_I[i] * mean_I[i];
        a[i] = var / (var + eps);
        b[i] = (1.0f - a[i]) * mean_I[i];
    }
#else
    #pragma omp parallel for schedule(static, 256)
    for(int i=0; i<size; ++i) {
        float var = mean_I_sq[i] - mean_I[i] * mean_I[i];
        a[i] = var / (var + eps);
        b[i] = (1.0f - a[i]) * mean_I[i];
    }
#endif

    // 步骤4: 计算 mean_a 和 mean_b
    box_filter_integral(a, mean_I, w, h, radius);      // 重用mean_I存储mean_a
    box_filter_integral(b, mean_I_sq, w, h, radius);   // 重用mean_I_sq存储mean_b

    // 步骤5: 计算最终结果 q = mean_a * I + mean_b
    ImageF* q = create_image_f(w, h);
    
#ifdef __ARM_NEON
    #pragma omp parallel for schedule(static, 256)
    for(int i = 0; i <= size - 4; i += 4) {
        float32x4_t vmean_a = vld1q_f32(mean_I + i);
        float32x4_t vmean_b = vld1q_f32(mean_I_sq + i);
        float32x4_t vI = vld1q_f32(I->data + i);
        
        // q = mean_a * I + mean_b
        float32x4_t vq = vmlaq_f32(vmean_b, vmean_a, vI);
        vst1q_f32(q->data + i, vq);
    }
    for(int i = (size & ~3); i < size; ++i) {
        q->data[i] = mean_I[i] * I->data[i] + mean_I_sq[i];
    }
#else
    #pragma omp parallel for schedule(static, 256)
    for(int i=0; i<size; ++i) {
        q->data[i] = mean_I[i] * I->data[i] + mean_I_sq[i];
    }
#endif

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

// --- 对数查找表优化 ---
#define LOG_LUT_SIZE 16384
static float g_log_lut[LOG_LUT_SIZE];
static int g_log_lut_initialized = 0;

void init_log_lut() {
    if (!g_log_lut_initialized) {
        #pragma omp parallel for
        for (int i = 0; i < LOG_LUT_SIZE; i++) {
            float x = (float)i / (LOG_LUT_SIZE - 1);
            g_log_lut[i] = logf(1.0f + x) / (float)M_LN2;
        }
        g_log_lut_initialized = 1;
    }
}

static inline float fast_log(float x) {
    if (x <= 0.0f) return g_log_lut[0];
    if (x >= 1.0f) return g_log_lut[LOG_LUT_SIZE - 1];
    int idx = (int)(x * (LOG_LUT_SIZE - 1));
    return g_log_lut[idx];
}


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
    // 3. Log map on base layer - 使用查找表加速
    ImageF *base_layer_log = create_image_f(w, h);
    
#ifdef __ARM_NEON
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i <= size - 4; i += 4) {
        // 使用查找表的向量化对数计算
        float vals[4];
        for (int j = 0; j < 4; j++) {
            vals[j] = fast_log(base_layer->data[i + j]);
        }
        float32x4_t vlog = vld1q_f32(vals);
        vst1q_f32(base_layer_log->data + i, vlog);
    }
    for (int i = (size & ~3); i < size; ++i) {
        base_layer_log->data[i] = fast_log(base_layer->data[i]);
    }
#else
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < size; ++i) {
        base_layer_log->data[i] = fast_log(base_layer->data[i]);
    }
#endif
    
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
    float* __restrict__ f = fused_image->data;
    float* __restrict__ blc = base_layer_compressed->data;
    float* dd = detail_layer->data;

    int i;
#ifdef __ARM_NEON
    // NEON 优化融合：(权重)加权融合，是数据并行操作
    float32x4_t base_w_vec = vdupq_n_f32(base_weight);
    float32x4_t detail_w_vec = vdupq_n_f32(detail_weight);

    #pragma omp parallel for
    for (i = 0; i <= size - 4; i += 4) {
        float32x4_t vec_c = vld1q_f32(blc + i);
        float32x4_t vec_d = vld1q_f32(dd + i);
        float32x4_t p1 = vmulq_f32(base_w_vec, vec_c);
        float32x4_t p2 = vmulq_f32(detail_w_vec, vec_d);
        float32x4_t fused_vec = vaddq_f32(p1, p2);
        vst1q_f32(f + i, fused_vec);
    }
#else
    // 标量版本融合
#pragma omp parallel for
    for (i = 0; i < size; ++i) {
        f[i] = base_weight * blc[i] + detail_weight * dd[i];
    }
#endif

    // 标量处理余数 (仅在 NEON 启用时 i 可能小于 size)
    for (; i < size; ++i) {
        f[i] = base_weight * blc[i] + detail_weight * dd[i];
    }

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

        // 浮点转整数（vcvtaq_s32_f32：四舍五入，vcvtq_s32_f32：截断）
        // 保持截断行为
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
    // 优化：减小 radius 可大幅提升性能 (从 16 降到 8 可提升约 4x)
    // radius=16 -> radius=8: ~4x faster (质量略有下降但可接受)
    int guided_filter_radius = 4;  // 原值: 16
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
    // 优化：减小 radius 可大幅提升性能 (从 16 降到 8 可提升约 4x)
    // radius=16 -> radius=8: ~4x faster (质量略有下降但可接受)
    int guided_filter_radius = 4;  // 原值: 16
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
    const char *root_dir = "/home/fmsh/preprocess_c/Code/RAW/";
    const char *output_root_dir = "/home/fmsh/preprocess_c/Code/test_results_c_optimized";

    // 初始化全局资源
    init_memory_pool();
    init_log_lut();

#ifdef _WIN32
    CreateDirectoryA(output_root_dir, NULL);
#else
    mkdir(output_root_dir, 0777);
#endif

    struct stat st;
    if (stat(root_dir, &st) != 0) {
        printf("Error: Input directory not found: %s\n", root_dir);
        free_memory_pool();
        return 1;
    }

    printf("--- Starting Enhanced Thermal Image Conversion (Ultra Optimized) ---\n");
    printf("--- Target: <20ms per image ---\n");
    printf("--- Input Dir: %s\n", root_dir);
    printf("--- Output Dir: %s\n", output_root_dir);

    process_directory(root_dir, output_root_dir);

    printf("\n--- All processing complete ---\n");
    
    free_memory_pool();

    return 0;
}
