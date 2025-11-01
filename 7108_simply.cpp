// ARM Cortex-A53 简化版本 - 热成像图像转换算法
// 删除了guided filter等耗时操作,只保留核心功能
// 
// 编译命令:
// gcc -march=native -O3 -fopenmp -ffast-math thermal_simple.c -o thermal_converter -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
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

// --- Memory Management ---
ImageF* create_image_f(int width, int height) {
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
    int is_dbout = (strstr(fname, "DBOut") != NULL);

    // NEON优化的数据转换
    #pragma omp parallel for schedule(static) num_threads(ARM_CORTEX_A53_CORES)
    for (int i = 0; i <= n_pixels_in_frame - 8; i += 8) {
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
        
        vst1q_f32(&img_out->data[i], low_f);
        vst1q_f32(&img_out->data[i + 4], high_f);
    }
    
    // 处理余数
    for (int i = (n_pixels_in_frame & ~7); i < n_pixels_in_frame; ++i) {
        uint16_t val = raw_data[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8);
        }
        img_out->data[i] = (float)(val & 0x3FFF);
    }

    free(raw_data);
    return img_out;
}

void write_bmp_grayscale(const char *fname, const ImageU8 *img) {
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
    info_header[32] = (unsigned char)(256);
    info_header[33] = (unsigned char)(256 >> 8);

    fwrite(file_header, 1, file_header_size, f);
    fwrite(info_header, 1, info_header_size, f);

    unsigned char palette[1024];
    for(int i = 0; i < 256; i++) {
        palette[i*4 + 0] = i;
        palette[i*4 + 1] = i;
        palette[i*4 + 2] = i;
        palette[i*4 + 3] = 0;
    }
    fwrite(palette, 1, palette_size, f);

    unsigned char *row_buffer = (unsigned char*)malloc(row_padded);
    for (int i = 0; i < h; i++) {
        memcpy(row_buffer, img->data + (h - 1 - i) * w, w);
        fwrite(row_buffer, 1, row_padded, f);
    }

    free(row_buffer);
    fclose(f);
}

// --- 简化的百分位数计算 (使用直方图) ---
float get_percentile_fast(ImageF *img, float p) {
    int size = img->width * img->height;
    
    // 找到min/max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) num_threads(ARM_CORTEX_A53_CORES)
    for (int i = 0; i < size; i++) {
        if (img->data[i] < min_val) min_val = img->data[i];
        if (img->data[i] > max_val) max_val = img->data[i];
    }
    
    if (max_val - min_val < 1e-6f) return min_val;
    
    // 构建直方图
    #define HIST_BINS 2048
    int histogram[HIST_BINS] = {0};
    float range = max_val - min_val;
    float scale = (HIST_BINS - 1) / range;
    
    #pragma omp parallel num_threads(ARM_CORTEX_A53_CORES)
    {
        int local_hist[HIST_BINS] = {0};
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < size; ++i) {
            int bin = (int)((img->data[i] - min_val) * scale);
            if (bin < 0) bin = 0;
            if (bin >= HIST_BINS) bin = HIST_BINS - 1;
            local_hist[bin]++;
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < HIST_BINS; ++i) {
                histogram[i] += local_hist[i];
            }
        }
    }
    
    // 找到对应百分位的bin
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
    
    float result = min_val + target_bin / scale;
    return result;
    #undef HIST_BINS
}

// --- 简化的转换算法 (删除guided filter, 只用基础归一化) ---
ImageU8* simple_thermal_conversion(ImageF* image_14bit) {
    int w = image_14bit->width;
    int h = image_14bit->height;
    int size = w * h;

    // 1. 计算百分位数用于对比度拉伸
    float min_val = get_percentile_fast(image_14bit, 0.5f);
    float max_val = get_percentile_fast(image_14bit, 99.5f);
    
    float range = max_val - min_val;
    if (range <= 1e-6f) range = 1.0f;

    // 2. 直接归一化到8-bit (NEON优化)
    ImageU8 *image_8bit = create_image_u8(w, h);
    float scale = 255.0f / range;

    float32x4_t min_v = vdupq_n_f32(min_val);
    float32x4_t scale_v = vdupq_n_f32(scale);
    float32x4_t v255 = vdupq_n_f32(255.0f);
    float32x4_t v0 = vdupq_n_f32(0.0f);

    #pragma omp parallel for schedule(static) num_threads(ARM_CORTEX_A53_CORES)
    for (int k = 0; k <= size - 8; k += 8) {
        float32x4_t val_low = vld1q_f32(image_14bit->data + k);
        float32x4_t val_high = vld1q_f32(image_14bit->data + k + 4);

        // 归一化并缩放
        val_low = vsubq_f32(val_low, min_v);
        val_low = vmulq_f32(val_low, scale_v);
        val_high = vsubq_f32(val_high, min_v);
        val_high = vmulq_f32(val_high, scale_v);

        // 截断到 [0, 255]
        val_low = vmaxq_f32(val_low, v0);
        val_low = vminq_f32(val_low, v255);
        val_high = vmaxq_f32(val_high, v0);
        val_high = vminq_f32(val_high, v255);

        // 转换为uint8
        int32x4_t int_low = vcvtq_s32_f32(val_low);
        int32x4_t int_high = vcvtq_s32_f32(val_high);

        int16x8_t int_s16 = vcombine_s16(vmovn_s32(int_low), vmovn_s32(int_high));
        uint8x8_t final_u8 = vqmovun_s16(int_s16);

        vst1_u8(image_8bit->data + k, final_u8);
    }

    // 处理余数
    for (int k = (size & ~7); k < size; ++k) {
        float val = (image_14bit->data[k] - min_val) * scale;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        image_8bit->data[k] = (unsigned char)(int)val;
    }

    return image_8bit;
}

// --- Main Execution ---
void process_directory(const char *root_dir, const char *output_root_dir) {
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
            
            double start_time = omp_get_wtime();

            ImageF *thermal_img_14bit = read_raw_thermal(raw_path);
            if (!thermal_img_14bit) continue;

            // 检查是否全黑
            int size = thermal_img_14bit->width * thermal_img_14bit->height;
            float max_pix = 0.0f;
            
            #pragma omp parallel for reduction(max:max_pix) num_threads(ARM_CORTEX_A53_CORES)
            for (int i = 0; i < size; ++i) {
                if (thermal_img_14bit->data[i] > max_pix) {
                    max_pix = thermal_img_14bit->data[i];
                }
            }

            ImageU8 *final_image;
            if (max_pix == 0) {
                final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
            } else {
                final_image = simple_thermal_conversion(thermal_img_14bit);
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
            printf("Done (%.3f ms)\n", (end_time - start_time) * 1000.0);

            free_image_f(thermal_img_14bit);
            free_image_u8(final_image);
        }
    }
    closedir(d);
}

int main() {
    const char *root_dir = "/home/fmsh/preprocess_c/Code/RAW/";
    const char *output_root_dir = "/home/fmsh/preprocess_c/Code/test_results_simple";

    omp_set_num_threads(ARM_CORTEX_A53_CORES);
    mkdir(output_root_dir, 0777);

    struct stat st;
    if (stat(root_dir, &st) != 0) {
        printf("Error: Input directory not found: %s\n", root_dir);
        return 1;
    }

    printf("=== Simplified ARM Thermal Image Converter ===\n");
    printf("Cores: %d | Input: %s\n", ARM_CORTEX_A53_CORES, root_dir);
    printf("Output: %s\n\n", output_root_dir);

    double total_start = omp_get_wtime();
    process_directory(root_dir, output_root_dir);
    double total_end = omp_get_wtime();

    printf("\n=== Complete (Total: %.3f s) ===\n", total_end - total_start);

    return 0;
}
