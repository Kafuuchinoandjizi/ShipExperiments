#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <float.h>
#include <stdint.h>

#define WIDTH 640
#define HEIGHT 512
#define EPS 1e-6f
#define MEM_POOL_SIZE 4

float g_mem_pool[MEM_POOL_SIZE][WIDTH * HEIGHT] __attribute__((aligned(16)));

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void box_filter(const float* src, float* dst, int w, int h, int radius) {
    if (radius <= 0) {
        memcpy(dst, src, w * h * sizeof(float));
        return;
    }

    int diameter = 2 * radius + 1;
    float inv_area = 1.0f / (diameter * diameter);
    float* temp = g_mem_pool[0];

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < h; i++) {
        const float* row_src = src + i * w;
        float* row_temp = temp + i * w;
        float sum = 0.0f;

        for (int k = 0; k <= radius && k < w; k++) {
            sum += row_src[k];
        }
        row_temp[0] = sum;

        for (int j = 1; j < w; j++) {
            if (j - radius - 1 >= 0) {
                sum -= row_src[j - radius - 1];
            }
            if (j + radius < w) {
                sum += row_src[j + radius];
            }
            row_temp[j] = sum;
        }
    }

#pragma omp parallel for num_threads(4)
    for (int j = 0; j < w; j++) {
        float* row_dst = dst + j;
        float sum = 0.0f;

        for (int k = 0; k <= radius && k < h; k++) {
            sum += temp[k * w + j];
        }
        row_dst[0] = sum * inv_area;

        for (int i = 1; i < h; i++) {
            if (i - radius - 1 >= 0) {
                sum -= temp[(i - radius - 1) * w + j];
            }
            if (i + radius < h) {
                sum += temp[(i + radius) * w + j];
            }
            row_dst[i * w] = sum * inv_area;
        }
    }
}

void guided_filter(const float* guided, const float* src, float* dst,
                   int w, int h, int radius, float eps) {
    int size = w * h;
    float *mean_I = g_mem_pool[1], *mean_p = g_mem_pool[2], *mean_Ip = g_mem_pool[3];
    float *var_I = g_mem_pool[0], *a = g_mem_pool[1], *b = g_mem_pool[2];
    float *mean_a = g_mem_pool[3], *mean_b = g_mem_pool[0];

    box_filter(guided, mean_I, w, h, radius);
    box_filter(src, mean_p, w, h, radius);

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        g_mem_pool[0][i] = guided[i] * src[i];
    }
    box_filter(g_mem_pool[0], mean_Ip, w, h, radius);

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        g_mem_pool[0][i] = guided[i] * guided[i];
    }
    box_filter(g_mem_pool[0], var_I, w, h, radius);

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        float mI = mean_I[i];
        float variance = var_I[i] - mI * mI;
        float cov_Ip = mean_Ip[i] - mI * mean_p[i];
        a[i] = cov_Ip / (variance + eps);
        b[i] = mean_p[i] - a[i] * mI;
    }

    box_filter(a, mean_a, w, h, radius);
    box_filter(b, mean_b, w, h, radius);

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        dst[i] = mean_a[i] * guided[i] + mean_b[i];
    }
}

void log_map(const float* src, float* dst, int size) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        dst[i] = logf(1.0f + src[i]);
    }
}

void fuse_layers(const float* base, const float* detail,
                 float base_weight, float detail_weight,
                 float* fused, int size) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        fused[i] = base[i] * base_weight + detail[i] * detail_weight;
    }
}

void normalize_to_8bit(const float* src, unsigned char* dst, int size) {
    float min_val = FLT_MAX, max_val = -FLT_MAX;
#pragma omp parallel reduction(min:min_val) reduction(max:max_val) num_threads(4)
    {
        float local_min = FLT_MAX, local_max = -FLT_MAX;
#pragma omp for
        for (int i = 0; i < size; i++) {
            local_min = fminf(local_min, src[i]);
            local_max = fmaxf(local_max, src[i]);
        }
        min_val = fminf(min_val, local_min);
        max_val = fmaxf(max_val, local_max);
    }

    float range = max_val - min_val;
    if (range < 1e-6f) range = 1e-6f;
    float scale = 255.0f / range;

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        float val = (src[i] - min_val) * scale;
        val = fmaxf(0.0f, fminf(255.0f, val));
        dst[i] = (unsigned char)(val);
    }
}

void guided_filter_pl_accelerator(const float* guided, const float* src, float* dst,
                                  int w, int h, int radius, float eps) {
    guided_filter(guided, src, dst, w, h, radius, eps);
}

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

    uint16_t* raw_buf = (uint16_t*)malloc(file_size);
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
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++) {
        uint16_t val = raw_buf[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8);
        }
        data[i] = (float)(val & 0x3FFF) / 0x3FFF;
    }

    free(raw_buf);
    return 0;
}

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

    unsigned char* row_buf = (unsigned char*)calloc(row_padded, 1);
    if (!row_buf) {
        perror("Failed to allocate BMP row buffer");
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < h; i++) {
        const unsigned char* row_data = data + (h - 1 - i) * w;
        memcpy(row_buf, row_data, w);
        fwrite(row_buf, 1, row_padded, fp);
    }

    free(row_buf);
    fclose(fp);
    return 0;
}

// 主函数：使用代码块隔离可能被goto跳过的变量
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.raw> <output.bmp>\n", argv[0]);
        return -1;
    }
    const char* input_path = argv[1];
    const char* output_path = argv[2];

    omp_set_num_threads(4);
    omp_set_dynamic(0);
    int size = WIDTH * HEIGHT;
    double total_start = get_time();

    // 内存分配：在所有goto之前完成
    float *image = (float*)aligned_alloc(16, size * sizeof(float));
    float *base_layer = (float*)aligned_alloc(16, size * sizeof(float));
    float *detail_layer = (float*)aligned_alloc(16, size * sizeof(float));
    float *fused_image = (float*)aligned_alloc(16, size * sizeof(float));
    unsigned char *bmp_data = (unsigned char*)malloc(size * sizeof(unsigned char));
    if (!image || !base_layer || !detail_layer || !fused_image || !bmp_data) {
        perror("Failed to allocate memory");
        goto cleanup;
    }

    // 步骤1：读取RAW文件（无被跳过的变量定义）
    {
        double start = get_time();
        if (read_raw_file(input_path, image, size) != 0) {
            fprintf(stderr, "Failed to process RAW file: %s\n", input_path);
            goto cleanup;
        }
        printf("[1/6] RAW 读取耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤2：计算Base层
    {
        double start = get_time();
        guided_filter_pl_accelerator(image, image, base_layer, WIDTH, HEIGHT, 8, EPS);
        printf("[2/6] Base 层计算耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤3：计算Detail层
    {
        double start = get_time();
#pragma omp parallel for num_threads(4)
        for (int i = 0; i < size; i++) {
            detail_layer[i] = image[i] - base_layer[i];
        }
        printf("[3/6] Detail 层计算耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤4：Log映射
    {
        double start = get_time();
        log_map(base_layer, base_layer, size);
        printf("[4/6] Log 映射耗时: %.3fms\n", (get_time() - start) * 1000);
    }

    // 步骤5：压缩融合（用代码块隔离const变量）
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

    // 总耗时统计（最后执行，不会被任何goto跳过）
    {
        double total_time = (get_time() - total_start) * 1000;
        printf("\n=== 总处理耗时: %.3fms ===\n", total_time);
    }

    cleanup:
    // 释放内存
    if (image) free(image);
    if (base_layer) free(base_layer);
    if (detail_layer) free(detail_layer);
    if (fused_image) free(fused_image);
    if (bmp_data) free(bmp_data);
    return 0;
}