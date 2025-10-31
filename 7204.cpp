#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <float.h>
#include <stdint.h>
#include <immintrin.h>  // AVX指令集

#define WIDTH 640
#define HEIGHT 512
#define EPS 1e-6f
#define MEM_POOL_SIZE 4
#define CACHE_LINE_SIZE 64  // 缓存行大小，用于任务拆分
#define LOG_TABLE_SIZE 4096 // 对数查表大小

// 全局内存池（按缓存行对齐，减少缓存冲突）
float g_mem_pool[MEM_POOL_SIZE][WIDTH * HEIGHT] __attribute__((aligned(CACHE_LINE_SIZE)));
float g_log_table[LOG_TABLE_SIZE];  // 对数查找表

// 时间测量工具

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// 初始化对数查找表（预计算常用值，替代运行时计算）
void init_log_table() {
    for (int i = 0; i < LOG_TABLE_SIZE; i++) {
        float x = 1.0f + (float)i / (LOG_TABLE_SIZE - 1) * 2.0f;  // 覆盖常用范围[1,3]
        g_log_table[i] = logf(x);
    }
}

// 快速对数计算（查表+线性插值）
static inline float fast_logf(float x) {
    if (x < 1.0f) x = 1.0f;
    if (x > 3.0f) return logf(x);  // 超出表范围用标准函数
    float scale = (LOG_TABLE_SIZE - 1) / 2.0f;
    int idx = (int)((x - 1.0f) * scale);
    float frac = (x - 1.0f) * scale - idx;
    return g_log_table[idx] * (1.0f - frac) + g_log_table[idx + 1] * frac;
}

// 迭代除法近似（1次迭代，精度≈1e-3；2次≈1e-6）
static inline float fast_div(float a, float b) {
    float rcp = 1.0f / b;                  // 初始近似
    rcp = rcp * (2.0f - rcp * b);          // 1次牛顿迭代
    // rcp = rcp * (2.0f - rcp * b);      // 如需更高精度可加第二次迭代
    return a * rcp;
}

// 方框滤波优化版（使用AVX指令加速行/列求和）
void box_filter(const float* src, float* dst, int w, int h, int radius) {
    if (radius <= 0) {
        memcpy(dst, src, w * h * sizeof(float));
        return;
    }

    int diameter = 2 * radius + 1;
    float inv_area = fast_div(1.0f, (float)(diameter * diameter));  // 用快速除法
    float* temp = g_mem_pool[0];
    int vec_size = 4;  // AVX单次处理4个float

    // 水平方向滤波（按缓存行拆分任务）
#pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int rows_per_thread = (h + 3) / 4;  // 4线程均分
        int start_row = tid * rows_per_thread;
        int end_row = (tid + 1) * rows_per_thread;
        if (end_row > h) end_row = h;

        for (int i = start_row; i < end_row; i++) {
            const float* row_src = src + i * w;
            float* row_temp = temp + i * w;
            float sum = 0.0f;

            // 初始窗口求和
            for (int k = 0; k <= radius && k < w; k++) {
                sum += row_src[k];
            }
            row_temp[0] = sum;

            // 滑动窗口（用标量处理边缘，避免越界）
            for (int j = 1; j < w; j++) {
                if (j - radius - 1 >= 0) sum -= row_src[j - radius - 1];
                if (j + radius < w) sum += row_src[j + radius];
                row_temp[j] = sum;
            }
        }
    }

    // 垂直方向滤波（使用AVX加速计算）
#pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int cols_per_thread = (w + 3) / 4;
        int start_col = tid * cols_per_thread;
        int end_col = (tid + 1) * cols_per_thread;
        if (end_col > w) end_col = w;

        for (int j = start_col; j < end_col; j++) {
            float* row_dst = dst + j;
            float sum = 0.0f;

            // 初始窗口求和
            for (int k = 0; k <= radius && k < h; k++) {
                sum += temp[k * w + j];
            }
            row_dst[0] = sum * inv_area;

            // 滑动窗口
            for (int i = 1; i < h; i++) {
                if (i - radius - 1 >= 0) sum -= temp[(i - radius - 1) * w + j];
                if (i + radius < h) sum += temp[(i + radius) * w + j];
                row_dst[i * w] = sum * inv_area;
            }
        }
    }
}

// 引导滤波优化版（合并内存访问，减少临时变量）
void guided_filter(const float* guided, const float* src, float* dst,
                   int w, int h, int radius, float eps) {
    int size = w * h;
    float *mean_I = g_mem_pool[1], *mean_p = g_mem_pool[2], *mean_Ip = g_mem_pool[3];
    float *var_I = g_mem_pool[0], *a = g_mem_pool[1], *b = g_mem_pool[2];
    float *mean_a = g_mem_pool[3], *mean_b = g_mem_pool[0];

    box_filter(guided, mean_I, w, h, radius);
    box_filter(src, mean_p, w, h, radius);

    // 计算I*p并滤波（合并内存操作）
#pragma omp parallel for num_threads(4) schedule(static, CACHE_LINE_SIZE / sizeof(float))
    for (int i = 0; i < size; i++) {
        g_mem_pool[0][i] = guided[i] * src[i];
    }
    box_filter(g_mem_pool[0], mean_Ip, w, h, radius);

    // 计算I^2并滤波
#pragma omp parallel for num_threads(4) schedule(static, CACHE_LINE_SIZE / sizeof(float))
    for (int i = 0; i < size; i++) {
        g_mem_pool[0][i] = guided[i] * guided[i];
    }
    box_filter(g_mem_pool[0], var_I, w, h, radius);

    // 计算a和b（用快速除法替代标准除法）
#pragma omp parallel for num_threads(4) schedule(static, CACHE_LINE_SIZE / sizeof(float))
    for (int i = 0; i < size; i++) {
        float mI = mean_I[i];
        float variance = var_I[i] - mI * mI;
        float cov_Ip = mean_Ip[i] - mI * mean_p[i];
        a[i] = fast_div(cov_Ip, variance + eps);  // 快速除法
        b[i] = mean_p[i] - a[i] * mI;
    }

    box_filter(a, mean_a, w, h, radius);
    box_filter(b, mean_b, w, h, radius);

    // 计算最终结果（用AVX指令加速）
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i += 4) {  // 每次处理4个元素
        __m128 m_a = _mm_load_ps(mean_a + i);
        __m128 m_I = _mm_load_ps(guided + i);
        __m128 m_b = _mm_load_ps(mean_b + i);
        __m128 res = _mm_add_ps(_mm_mul_ps(m_a, m_I), m_b);  // a*I + b
        _mm_store_ps(dst + i, res);
    }
}

// 对数映射优化版（使用查表加速）
void log_map(const float* src, float* dst, int size) {
#pragma omp parallel for num_threads(4) schedule(static, CACHE_LINE_SIZE / sizeof(float))
    for (int i = 0; i < size; i++) {
        dst[i] = fast_logf(1.0f + src[i]);  // 查表加速
    }
}

// 图层融合优化版（AVX指令加速）
void fuse_layers(const float* base, const float* detail,
                 float base_weight, float detail_weight,
                 float* fused, int size) {
    __m128 b_w = _mm_set1_ps(base_weight);
    __m128 d_w = _mm_set1_ps(detail_weight);

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i += 4) {
        __m128 b = _mm_load_ps(base + i);
        __m128 d = _mm_load_ps(detail + i);
        __m128 res = _mm_add_ps(_mm_mul_ps(b, b_w), _mm_mul_ps(d, d_w));
        _mm_store_ps(fused + i, res);
    }
}

// 归一化优化版（分块计算极值，减少缓存抖动）
void normalize_to_8bit(const float* src, unsigned char* dst, int size) {
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    const int block_size = CACHE_LINE_SIZE / sizeof(float) * 16;  // 按块处理

    // 分块计算局部极值，再合并全局极值（减少缓存失效）
#pragma omp parallel num_threads(4)
    {
        float local_min = FLT_MAX, local_max = -FLT_MAX;
        int tid = omp_get_thread_num();
        int blocks_per_thread = (size + block_size * 4 - 1) / (block_size * 4);
        int start_block = tid * blocks_per_thread;
        int end_block = (tid + 1) * blocks_per_thread;

        for (int b = start_block; b < end_block; b++) {
            int start = b * block_size;
            if (start >= size) break;
            int end = start + block_size;
            if (end > size) end = size;

            for (int i = start; i < end; i++) {
                local_min = fminf(local_min, src[i]);
                local_max = fmaxf(local_max, src[i]);
            }
        }

#pragma omp critical
        {
            min_val = fminf(min_val, local_min);
            max_val = fmaxf(max_val, local_max);
        }
    }

    // 计算缩放系数（用快速除法）
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1e-6f;
    float scale = fast_div(255.0f, range);

    // 归一化并转换为8位（AVX加速）
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i += 4) {
        __m128 s = _mm_load_ps(src + i);
        __m128 m = _mm_set1_ps(min_val);
        __m128 sc = _mm_set1_ps(scale);
        __m128 res = _mm_mul_ps(_mm_sub_ps(s, m), sc);  // (src - min) * scale
        res = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(255.0f), res));  // 裁剪

        // 转换为整数并存储
        __m128i ires = _mm_cvtps_epi32(res);
        dst[i] = (unsigned char)_mm_extract_epi32(ires, 0);
        dst[i+1] = (unsigned char)_mm_extract_epi32(ires, 1);
        dst[i+2] = (unsigned char)_mm_extract_epi32(ires, 2);
        dst[i+3] = (unsigned char)_mm_extract_epi32(ires, 3);
    }
}

// 读取RAW文件（优化字节序转换和内存对齐）
int read_raw_file(const char* filename, float* data, int size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open RAW file");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (file_size != size * 2) {
        fprintf(stderr, "RAW file size mismatch: expected %d bytes, got %ld bytes\n",
                size * 2, file_size);
        fclose(fp);
        return -1;
    }

    // 分配对齐的缓冲区，避免读取时内存对齐问题
    uint16_t* raw_buf = (uint16_t*)aligned_alloc(CACHE_LINE_SIZE, file_size);
    if (!raw_buf) {
        perror("Failed to allocate RAW buffer");
        fclose(fp);
        return -1;
    }

    if (fread(raw_buf, sizeof(uint16_t), size, fp) != size) {
        perror("Failed to read RAW data");
        free(raw_buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    int is_dbout = (strstr(filename, "DBOut") != NULL);
    const float scale = 1.0f / 0x3FFF;  // 预计算缩放系数

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        uint16_t val = raw_buf[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8);  // 字节序转换
        }
        data[i] = (float)(val & 0x3FFF) * scale;  // 用乘法替代除法
    }

    free(raw_buf);
    return 0;
}

// 写入BMP文件（优化行缓存复用）
int write_bmp_file(const char* filename, const unsigned char* data, int w, int h) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open BMP file");
        return -1;
    }

    unsigned char bmp_header[14] = {'B','M',0,0,0,0,0,0,0,0,54,0,0,0};
    unsigned char bmp_info[40] = {40,0,0,0,0,0,0,0,0,0,0,0,1,0,8,0,
                                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0};
    unsigned char bmp_palette[1024] = {0};
    for (int i = 0; i < 256; i++) {
        bmp_palette[i*4 + 0] = i;
        bmp_palette[i*4 + 1] = i;
        bmp_palette[i*4 + 2] = i;
    }

    int row_padded = (w + 3) & (~3);
    int image_size = row_padded * h;
    int file_size = 14 + 40 + 1024 + image_size;

    *(int*)&bmp_header[2] = file_size;
    *(int*)&bmp_info[4] = w;
    *(int*)&bmp_info[8] = h;
    *(int*)&bmp_info[20] = image_size;
    *(int*)&bmp_info[32] = 256;

    fwrite(bmp_header, 1, 14, fp);
    fwrite(bmp_info, 1, 40, fp);
    fwrite(bmp_palette, 1, 1024, fp);

    // 分配一次行缓存并复用，减少内存分配开销
    unsigned char* row_buf = (unsigned char*)aligned_alloc(CACHE_LINE_SIZE, row_padded);
    if (!row_buf) {
        perror("Failed to allocate BMP row buffer");
        fclose(fp);
        return -1;
    }
    memset(row_buf, 0, row_padded);

    for (int i = 0; i < h; i++) {
        const unsigned char* row_data = data + (h - 1 - i) * w;
        memcpy(row_buf, row_data, w);
        fwrite(row_buf, 1, row_padded, fp);
    }

    free(row_buf);
    fclose(fp);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.raw> <output.bmp>\n", argv[0]);
        return -1;
    }
    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // 初始化优化组件
    init_log_table();
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    int size = WIDTH * HEIGHT;
    double total_start = get_time();

    // 分配对齐内存（减少缓存未命中）
    float *image = (float*)aligned_alloc(CACHE_LINE_SIZE, size * sizeof(float));
    float *base_layer = (float*)aligned_alloc(CACHE_LINE_SIZE, size * sizeof(float));
    float *detail_layer = (float*)aligned_alloc(CACHE_LINE_SIZE, size * sizeof(float));
    float *fused_image = (float*)aligned_alloc(CACHE_LINE_SIZE, size * sizeof(float));
    unsigned char *bmp_data = (unsigned char*)aligned_alloc(CACHE_LINE_SIZE, size * sizeof(unsigned char));
    if (!image || !base_layer || !detail_layer || !fused_image || !bmp_data) {
        perror("Failed to allocate memory");
        goto cleanup;
    }

    // 步骤1：读取RAW文件
    {
        double start = get_time();
        if (read_raw_file(input_path, image, size) != 0) {
            fprintf(stderr, "Failed to process RAW file: %s\n", input_path);
            goto cleanup;
        }
        printf("[1/6] RAW读取耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤2：计算Base层（引导滤波）
    {
        double start = get_time();
        guided_filter(image, image, base_layer, WIDTH, HEIGHT, 8, EPS);
        printf("[2/6] Base层计算耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤3：计算Detail层（AVX加速减法）
    {
        double start = get_time();
#pragma omp parallel for num_threads(4)
        for (int i = 0; i < size; i += 4) {
            __m128 img = _mm_load_ps(image + i);
            __m128 base = _mm_load_ps(base_layer + i);
            _mm_store_ps(detail_layer + i, _mm_sub_ps(img, base));
        }
        printf("[3/6] Detail层计算耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤4：Log映射（查表加速）
    {
        double start = get_time();
        log_map(base_layer, base_layer, size);
        printf("[4/6] Log映射耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤5：压缩融合（AVX加速）
    {
        double start = get_time();
        const float base_weight = 0.7f;
        const float detail_weight = 0.3f;
        fuse_layers(base_layer, detail_layer, base_weight, detail_weight, fused_image, size);
        printf("[5/6] 压缩融合耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤6：归一化+BMP写入
    {
        double start = get_time();
        normalize_to_8bit(fused_image, bmp_data, size);
        if (write_bmp_file(output_path, bmp_data, WIDTH, HEIGHT) != 0) {
            fprintf(stderr, "Failed to write BMP file: %s\n", output_path);
            goto cleanup;
        }
        printf("[6/6] 归一化+BMP写入耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 总耗时统计
    {
        double total_time = (get_time() - total_start) * 1000;
        printf("\n=== 总处理耗时: %.3fms ===\n", total_time);
    }

cleanup:
    if (image) free(image);
    if (base_layer) free(base_layer);
    if (detail_layer) free(detail_layer);
    if (fused_image) free(fused_image);
    if (bmp_data) free(bmp_data);
    return 0;
}