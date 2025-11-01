// ARM Cortex-A53 极简版本 - 热成像图像转换算法
// 删除百分位数计算,使用固定范围归一化,目标<5ms
//
// 编译命令:
// aarch64-linux-gnu-g++ -march=armv8-a+simd -mtune=cortex-a53 -O3 -fopenmp -ffast-math -ftree-vectorize -funroll-loops thermal_simple.cpp -o thermal_converter -lm -lpthread

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <float.h>

#define __ARM_NEON 1
#include <arm_neon.h>
#include <omp.h>

#define ARM_CORTEX_A53_CORES 4
#define MAX_WIDTH 640
#define MAX_HEIGHT 512
#define MAX_PIXELS (MAX_WIDTH * MAX_HEIGHT)

// 全局缓冲区(避免重复分配)
static float g_buffer[MAX_PIXELS] __attribute__((aligned(64)));
static unsigned char g_output[MAX_PIXELS] __attribute__((aligned(64)));

// --- Data Structures ---
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

// --- 文件大小查询 ---
long get_file_size(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

// --- NEON优化的min/max查找 ---
static inline void neon_find_min_max(const float* data, int size, float* min_val, float* max_val) {
    float32x4_t vmin = vdupq_n_f32(FLT_MAX);
    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);

    int i;
    for (i = 0; i <= size - 16; i += 16) {
        float32x4_t v1 = vld1q_f32(data + i);
        float32x4_t v2 = vld1q_f32(data + i + 4);
        float32x4_t v3 = vld1q_f32(data + i + 8);
        float32x4_t v4 = vld1q_f32(data + i + 12);

        vmin = vminq_f32(vmin, v1);
        vmin = vminq_f32(vmin, v2);
        vmin = vminq_f32(vmin, v3);
        vmin = vminq_f32(vmin, v4);

        vmax = vmaxq_f32(vmax, v1);
        vmax = vmaxq_f32(vmax, v2);
        vmax = vmaxq_f32(vmax, v3);
        vmax = vmaxq_f32(vmax, v4);
    }

    // 水平归约
    float min_arr[4], max_arr[4];
    vst1q_f32(min_arr, vmin);
    vst1q_f32(max_arr, vmax);

    *min_val = min_arr[0];
    *max_val = max_arr[0];
    for (int j = 1; j < 4; j++) {
        if (min_arr[j] < *min_val) *min_val = min_arr[j];
        if (max_arr[j] > *max_val) *max_val = max_arr[j];
    }

    // 处理余数
    for (; i < size; ++i) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

// --- 读取RAW文件(优化版) ---
int read_raw_thermal(const char *fname, float* out_data, int* out_w, int* out_h) {
    long file_size = get_file_size(fname);
    if (file_size <= 0) return 0;

    FILE *f = fopen(fname, "rb");
    if (!f) return 0;

    long num_pixels = file_size / sizeof(uint16_t);
    uint16_t *raw_data = (uint16_t*)malloc(file_size);
    if (!raw_data) {
        fclose(f);
        return 0;
    }

    if (fread(raw_data, sizeof(uint16_t), num_pixels, f) != (size_t)num_pixels) {
        free(raw_data);
        fclose(f);
        return 0;
    }
    fclose(f);

    int w = 640, h = 512;
    if (num_pixels < 320 * 256) {
        w = 192; h = 160;
    } else if (num_pixels < 640 * 512) {
        w = 320; h = 256;
    }

    *out_w = w;
    *out_h = h;
    int n_pixels = w * h;
    int is_dbout = (strstr(fname, "DBOut") != NULL);

    // NEON优化转换
#pragma omp parallel for schedule(static) num_threads(ARM_CORTEX_A53_CORES)
    for (int i = 0; i <= n_pixels - 8; i += 8) {
        uint16x8_t val = vld1q_u16(&raw_data[i]);

        if (is_dbout) {
            uint16x8_t val_high = vshrq_n_u16(val, 8);
            uint16x8_t val_low = vshlq_n_u16(val, 8);
            val = vorrq_u16(val_high, val_low);
        }

        uint16x8_t mask = vdupq_n_u16(0x3FFF);
        uint16x8_t masked = vandq_u16(val, mask);

        uint32x4_t low32 = vmovl_u16(vget_low_u16(masked));
        uint32x4_t high32 = vmovl_u16(vget_high_u16(masked));

        float32x4_t low_f = vcvtq_f32_u32(low32);
        float32x4_t high_f = vcvtq_f32_u32(high32);

        vst1q_f32(&out_data[i], low_f);
        vst1q_f32(&out_data[i + 4], high_f);
    }

    // 处理余数
    for (int i = (n_pixels & ~7); i < n_pixels; ++i) {
        uint16_t val = raw_data[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8);
        }
        out_data[i] = (float)(val & 0x3FFF);
    }

    free(raw_data);
    return 1;
}

// --- 写BMP文件 ---
void write_bmp_grayscale(const char *fname, const unsigned char* data, int w, int h) {
    FILE *f = fopen(fname, "wb");
    if (!f) return;

    int row_padded = (w + 3) & (~3);
    int image_size = row_padded * h;
    int palette_size = 256 * 4;
    uint32_t off_bits = 14 + 40 + palette_size;
    uint32_t file_size = off_bits + image_size;

    unsigned char file_header[14] = {'B', 'M'};
    file_header[2] = (unsigned char)(file_size);
    file_header[3] = (unsigned char)(file_size >> 8);
    file_header[4] = (unsigned char)(file_size >> 16);
    file_header[5] = (unsigned char)(file_size >> 24);
    file_header[10] = (unsigned char)(off_bits);
    file_header[11] = (unsigned char)(off_bits >> 8);
    file_header[12] = (unsigned char)(off_bits >> 16);
    file_header[13] = (unsigned char)(off_bits >> 24);

    unsigned char info_header[40] = {0};
    info_header[0] = 40;
    info_header[4] = (unsigned char)(w);
    info_header[5] = (unsigned char)(w >> 8);
    info_header[6] = (unsigned char)(w >> 16);
    info_header[7] = (unsigned char)(w >> 24);
    info_header[8] = (unsigned char)(h);
    info_header[9] = (unsigned char)(h >> 8);
    info_header[10] = (unsigned char)(h >> 16);
    info_header[11] = (unsigned char)(h >> 24);
    info_header[12] = 1;
    info_header[14] = 8;
    info_header[20] = (unsigned char)(image_size);
    info_header[21] = (unsigned char)(image_size >> 8);
    info_header[22] = (unsigned char)(image_size >> 16);
    info_header[23] = (unsigned char)(image_size >> 24);
    info_header[32] = 0;
    info_header[33] = 1;

    fwrite(file_header, 1, 14, f);
    fwrite(info_header, 1, 40, f);

    unsigned char palette[1024];
    for(int i = 0; i < 256; i++) {
        palette[i*4] = i;
        palette[i*4+1] = i;
        palette[i*4+2] = i;
        palette[i*4+3] = 0;
    }
    fwrite(palette, 1, palette_size, f);

    unsigned char *row_buffer = (unsigned char*)malloc(row_padded);
    for (int i = 0; i < h; i++) {
        memcpy(row_buffer, data + (h - 1 - i) * w, w);
        fwrite(row_buffer, 1, row_padded, f);
    }

    free(row_buffer);
    fclose(f);
}

// --- 超快速转换(无百分位数计算) ---
void ultrafast_thermal_conversion(const float* src, unsigned char* dst, int size, const char* debug_folder) {

    char debug_path[256];

    // 并行查找min/max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

#pragma omp parallel num_threads(ARM_CORTEX_A53_CORES)
    {
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;

#pragma omp for schedule(static) nowait
        for (int chunk = 0; chunk < ARM_CORTEX_A53_CORES; chunk++) {
            int start = (size * chunk) / ARM_CORTEX_A53_CORES;
            int end = (size * (chunk + 1)) / ARM_CORTEX_A53_CORES;
            float thread_min, thread_max;
            neon_find_min_max(src + start, end - start, &thread_min, &thread_max);
            if (thread_min < local_min) local_min = thread_min;
            if (thread_max > local_max) local_max = thread_max;
        }

#pragma omp critical
        {
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
        }
    }

    // 使用1%-99%范围(简单裁剪避免极值)
    float range_full = max_val - min_val;
    float trim_amount = range_full * 0.01f;
    min_val += trim_amount;
    max_val -= trim_amount;

    float range = max_val - min_val;
    if (range <= 1e-6f) range = 1.0f;
    float scale = 255.0f / range;

    // NEON批量转换
    float32x4_t min_v = vdupq_n_f32(min_val);
    float32x4_t scale_v = vdupq_n_f32(scale);
    float32x4_t v255 = vdupq_n_f32(255.0f);
    float32x4_t v0 = vdupq_n_f32(0.0f);

#pragma omp parallel for schedule(static) num_threads(ARM_CORTEX_A53_CORES)
    for (int k = 0; k <= size - 16; k += 16) {
        // 处理16个像素
        float32x4_t v1 = vld1q_f32(src + k);
        float32x4_t v2 = vld1q_f32(src + k + 4);
        float32x4_t v3 = vld1q_f32(src + k + 8);
        float32x4_t v4 = vld1q_f32(src + k + 12);

        // 归一化
        v1 = vmulq_f32(vsubq_f32(v1, min_v), scale_v);
        v2 = vmulq_f32(vsubq_f32(v2, min_v), scale_v);
        v3 = vmulq_f32(vsubq_f32(v3, min_v), scale_v);
        v4 = vmulq_f32(vsubq_f32(v4, min_v), scale_v);

        // 截断
        v1 = vmaxq_f32(vminq_f32(v1, v255), v0);
        v2 = vmaxq_f32(vminq_f32(v2, v255), v0);
        v3 = vmaxq_f32(vminq_f32(v3, v255), v0);
        v4 = vmaxq_f32(vminq_f32(v4, v255), v0);

        // 转换为uint8
        uint16x8_t u16_1 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(v1)), vqmovn_u32(vcvtq_u32_f32(v2)));
        uint16x8_t u16_2 = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(v3)), vqmovn_u32(vcvtq_u32_f32(v4)));

        uint8x16_t u8_final = vcombine_u8(vqmovn_u16(u16_1), vqmovn_u16(u16_2));
        vst1q_u8(dst + k, u8_final);
    }

    // 处理余数
    for (int k = (size & ~15); k < size; ++k) {
        float val = (src[k] - min_val) * scale;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        dst[k] = (unsigned char)(int)val;
    }

    sprintf(debug_path, "%s/c_10_final_8bit.txt", debug_folder);
    save_image_u8_to_txt(image_8bit, debug_path);

}

// --- 主处理函数 ---
void process_directory(const char *root_dir, const char *output_root_dir, const char *debug_dir) {
    DIR *d = opendir(root_dir);
    if (!d) {
        printf("Error: Input directory not found %s\n", root_dir);
        return;
    }

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        const char *ext = strrchr(dir->d_name, '.');
        if (ext && strcmp(ext, ".raw") == 0) {
            char raw_path[1024];
            snprintf(raw_path, sizeof(raw_path), "%s/%s", root_dir, dir->d_name);

            printf("Processing: %s ... ", dir->d_name);
            fflush(stdout);

            char stem[256];
            char image_debug_dir[1024];
            snprintf(image_debug_dir, sizeof(image_debug_dir), "%s/%s", debug_dir, stem);
            mkdir(image_debug_dir, 0777);
            printf("Debug files will be saved in: %s\\n", image_debug_dir);
            ImageF *thermal_img_14bit = read_raw_thermal(raw_path, image_debug_dir);

            double start_time = omp_get_wtime();

            int w, h;
            if (!read_raw_thermal(raw_path, g_buffer, &w, &h)) {
                printf("Failed\n");
                continue;
            }

            int size = w * h;

            // 检查是否全黑
            float max_pix = 0.0f;
#pragma omp parallel for reduction(max:max_pix) num_threads(ARM_CORTEX_A53_CORES)
            for (int i = 0; i < size; ++i) {
                if (g_buffer[i] > max_pix) max_pix = g_buffer[i];
            }

            if (max_pix == 0) {
                memset(g_output, 0, size);
            } else {
                ultrafast_thermal_conversion(g_buffer, g_output, size, image_debug_dir);
            }

            char stem[256];
            strncpy(stem, dir->d_name, sizeof(stem));
            char *dot = strrchr(stem, '.');
            if (dot) *dot = '\0';

            char output_path[1024];
            snprintf(output_path, sizeof(output_path), "%s/%s.bmp", output_root_dir, stem);

            write_bmp_grayscale(output_path, g_output, w, h);

            double end_time = omp_get_wtime();
            printf("Done (%.3f ms)\n", (end_time - start_time) * 1000.0);
        }
    }
    closedir(d);
}

int main() {
    const char *root_dir = "/home/fmsh/preprocess_c/Code/RAW/";
    const char *output_root_dir = "/home/fmsh/preprocess_c/Code/test_results_ultrafast";
    const char *debug_dir = "debug_output";

    omp_set_num_threads(ARM_CORTEX_A53_CORES);
    mkdir(output_root_dir, 0777);
    mkdir(debug_dir, 0777);

    struct stat st;
    if (stat(root_dir, &st) != 0) {
        printf("Error: Input directory not found: %s\n", root_dir);
        return 1;
    }

    printf("=== Ultra-Fast ARM Thermal Converter ===\n");
    printf("Target: <5ms/frame | Cores: %d\n", ARM_CORTEX_A53_CORES);
    printf("Input: %s\nOutput: %s\n\n", root_dir, output_root_dir);

    double total_start = omp_get_wtime();
    process_directory(root_dir, output_root_dir, debug_dir);
    double total_end = omp_get_wtime();

    printf("\n=== Complete (Total: %.3f s) ===\n", total_end - total_start);

    return 0;
}