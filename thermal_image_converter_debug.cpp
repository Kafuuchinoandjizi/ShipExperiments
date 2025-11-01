// A C implementation of the thermal image conversion algorithm from the Python script.
// DEBUG VERSION: This version saves intermediate steps to text files for comparison.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <float.h>  // 添加这个头文件以使用DBL_MAX

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


// --- DEBUG: Function to save ImageF to a text file ---
void save_image_f_to_txt(const ImageF* img, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open file for writing");
        return;
    }
    for (int y = 0; y < img->height; ++y) {
        for (int x = 0; x < img->width; ++x) {
            fprintf(f, "%.8f ", img->data[y * img->width + x]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// --- DEBUG: Function to save ImageU8 to a text file ---
void save_image_u8_to_txt(const ImageU8* img, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open file for writing");
        return;
    }
    for (int y = 0; y < img->height; ++y) {
        for (int x = 0; x < img->width; ++x) {
            fprintf(f, "%d ", img->data[y * img->width + x]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}


// --- File I/O ---

long get_file_size(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return -1;
}

ImageF* read_raw_thermal(const char *fname, const char* debug_folder) {
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

    uint16_t *temp_row = (uint16_t *)malloc(w * sizeof(uint16_t));
    if(temp_row) {
        memcpy(temp_row, raw_data + w, w * sizeof(uint16_t));
        memcpy(raw_data, temp_row, w * sizeof(uint16_t));
        free(temp_row);
    }
    
    int is_dbout = (strstr(fname, "DBOut") != NULL);

    for (int i = 0; i < n_pixels_in_frame; ++i) {
        uint16_t val = raw_data[i];
        if (is_dbout) {
            val = (val >> 8) | (val << 8);
        }
        img_out->data[i] = (float)(val & 0x3FFF);
    }
    
    // DEBUG: Save initial 14-bit data
    char debug_path[256];
    sprintf(debug_path, "%s/c_0_initial_14bit.txt", debug_folder);
    save_image_f_to_txt(img_out, debug_path);

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
        fwrite(row_buffer, 1, row_padded, f);
    }

    free(row_buffer);
    fclose(f);
}

// --- Algorithm Implementation ---

// Helper for qsort
int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

float get_percentile(ImageF *img, float p) {
    int size = img->width * img->height;
    float *sorted_data = (float*)malloc(size * sizeof(float));
    if (!sorted_data) {
        fprintf(stderr, "Failed to allocate memory for percentile calculation.\n");
        return 0.0f;
    }
    memcpy(sorted_data, img->data, size * sizeof(float));
    qsort(sorted_data, size, sizeof(float), compare_floats);

    int index = (int)(((size_t)size - 1) * p / 100.0);
    float result = sorted_data[index];
    free(sorted_data);
    return result;
}

// Implements BORDER_REFLECT_101 logic, as used in OpenCV's default implementation.
// E.g. for a length of 5 (indices 0,1,2,3,4), p=-1 maps to 1, p=5 maps to 3.
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

    // Vertical pass from src to temp, using double for accumulator
    for (int x = 0; x < w; ++x) {
        double sum = 0.0;
        // Initial sum for the first pixel in the column
        for (int y = -r; y <= r; ++y) {
            sum += src->data[border_reflect_101(y, h) * w + x];
        }
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


ImageF* guided_filter(const ImageF* src, int radius, float eps, const char* debug_folder) {
    int w = src->width;
    int h = src->height;
    int size = w * h;

    const ImageF *I = src; // In this case, guide and source are the same
    char debug_path[256];

    ImageF* mean_I = box_filter(I, radius);
    sprintf(debug_path, "%s/c_2_mean_I.txt", debug_folder);
    save_image_f_to_txt(mean_I, debug_path);


    ImageF* I_sq = create_image_f(w, h);
    if (!I_sq) { free_image_f(mean_I); return NULL; }
    for(int i=0; i<size; ++i) I_sq->data[i] = I->data[i] * I->data[i];

    ImageF* mean_I_sq = box_filter(I_sq, radius);
    
    // Variance is E[X^2] - (E[X])^2
    ImageF* var_I = create_image_f(w, h);
    if (!var_I) { free_image_f(mean_I); free_image_f(I_sq); free_image_f(mean_I_sq); return NULL; }
    for(int i=0; i<size; ++i) var_I->data[i] = mean_I_sq->data[i] - mean_I->data[i] * mean_I->data[i];
    sprintf(debug_path, "%s/c_3_var_I.txt", debug_folder);
    save_image_f_to_txt(var_I, debug_path);

    
    // In this specific case, p = I, so cov(I, p) = var(I).
    // a = cov(I,p) / (var(I) + eps) = var(I) / (var(I) + eps)
    ImageF* a = create_image_f(w, h);
    if (!a) { free_image_f(var_I); free_image_f(mean_I); free_image_f(I_sq); free_image_f(mean_I_sq); return NULL; }
    for(int i=0; i<size; ++i) a->data[i] = var_I->data[i] / (var_I->data[i] + eps);

    // b = mean_p - a * mean_I. Since p=I, b = mean_I - a * mean_I = (1-a) * mean_I
    ImageF* b = create_image_f(w, h);
    if (!b) { free_image_f(a); free_image_f(var_I); free_image_f(mean_I); free_image_f(I_sq); free_image_f(mean_I_sq); return NULL; }
    for(int i=0; i<size; ++i) b->data[i] = (1.0f - a->data[i]) * mean_I->data[i];

    ImageF* mean_a = box_filter(a, radius);
    ImageF* mean_b = box_filter(b, radius);

    // q = mean_a * I + mean_b
    ImageF* q = create_image_f(w, h);
    for(int i=0; i<size; ++i) q->data[i] = mean_a->data[i] * I->data[i] + mean_b->data[i];

    // Free intermediate images
    free_image_f(mean_I);
    free_image_f(I_sq);
    free_image_f(mean_I_sq);
    free_image_f(var_I);
    free_image_f(a);
    free_image_f(b);
    free_image_f(mean_a);
    free_image_f(mean_b);
    
    return q;
}

ImageU8* enhanced_thermal_conversion(
    ImageF* image_14bit,
    int radius,
    float eps,
    float min_percentile,
    float max_percentile,
    float base_weight,
    float detail_weight,
    const char* debug_folder)
{
    int w = image_14bit->width;
    int h = image_14bit->height;
    int size = w * h;
    char debug_path[256];

    // 1. Normalize to [0, 1]
    ImageF *image_normalized = create_image_f(w, h);
    for (int i = 0; i < size; ++i) {
        image_normalized->data[i] = image_14bit->data[i] / 16383.0f; // 2^14 - 1
    }
    sprintf(debug_path, "%s/c_1_normalized.txt", debug_folder);
    save_image_f_to_txt(image_normalized, debug_path);


    // 2. Guided filter
    ImageF *base_layer = guided_filter(image_normalized, radius, eps, debug_folder);
    sprintf(debug_path, "%s/c_4_base_layer.txt", debug_folder);
    save_image_f_to_txt(base_layer, debug_path);

    ImageF *detail_layer = create_image_f(w, h);
    for (int i = 0; i < size; ++i) {
        detail_layer->data[i] = image_normalized->data[i] - base_layer->data[i];
    }
    sprintf(debug_path, "%s/c_5_detail_layer.txt", debug_folder);
    save_image_f_to_txt(detail_layer, debug_path);

    
    // 3. Log map on base layer
    ImageF *base_layer_log = create_image_f(w, h);
    float c = 1.0f / logf(2.0f); // 1 / ln(2)
    for (int i = 0; i < size; ++i) {
        base_layer_log->data[i] = c * logf(1.0f + base_layer->data[i]);
    }
    sprintf(debug_path, "%s/c_6_base_layer_log.txt", debug_folder);
    save_image_f_to_txt(base_layer_log, debug_path);

    
    // 4. Compress and fuse
    float min_base_val = get_percentile(base_layer_log, min_percentile);
    float max_base_val = get_percentile(base_layer_log, max_percentile);
    
    ImageF *base_layer_compressed = create_image_f(w, h);
    for (int i = 0; i < size; ++i) {
        float val = base_layer_log->data[i];
        if (val < min_base_val) val = min_base_val;
        if (val > max_base_val) val = max_base_val;
        base_layer_compressed->data[i] = val;
    }
    sprintf(debug_path, "%s/c_7_base_layer_compressed.txt", debug_folder);
    save_image_f_to_txt(base_layer_compressed, debug_path);


    ImageF *fused_image = create_image_f(w, h);
    for (int i = 0; i < size; ++i) {
        fused_image->data[i] = base_weight * base_layer_compressed->data[i] +
                               detail_weight * detail_layer->data[i];
    }
    sprintf(debug_path, "%s/c_8_fused_image.txt", debug_folder);
    save_image_f_to_txt(fused_image, debug_path);


    // 5. Normalize to 8-bit
    // 使用更精确的方法计算min和max值
    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;
    
    // 单次遍历找出min和max，减少累积误差
    for (int i = 0; i < size; ++i) {
        double val = (double)fused_image->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    double range = max_val - min_val;
    if (range <= 1e-6) range = 1e-6;

    ImageU8 *image_8bit = create_image_u8(w, h);
    ImageF *image_8bit_float = create_image_f(w,h);

    // 尝试更接近NumPy的向量化操作
    // 预先计算系数，减少每个像素的计算量
    double scale = 255.0 / range;
    
    // 创建临时的double精度数组，模拟NumPy的向量化操作
    double *temp_buffer = (double*)malloc(size * sizeof(double));
    if (!temp_buffer) {
        // 内存分配失败，回退到原来的方法
        for (int i = 0; i < size; ++i) {
            double val = ((double)fused_image->data[i] - min_val) * scale;
            image_8bit_float->data[i] = (float)val;
        }
    } else {
        // 使用临时缓冲区进行更高精度的计算
        // 步骤1：减去min_val
        for (int i = 0; i < size; ++i) {
            temp_buffer[i] = (double)fused_image->data[i] - min_val;
        }
        
        // 步骤2：乘以scale
        for (int i = 0; i < size; ++i) {
            temp_buffer[i] *= scale;
        }
        
        // 步骤3：转换回float并存储
        for (int i = 0; i < size; ++i) {
            image_8bit_float->data[i] = (float)temp_buffer[i];
        }
        
        free(temp_buffer);
    }
    
    sprintf(debug_path, "%s/c_9_final_8bit_float.txt", debug_folder);
    save_image_f_to_txt(image_8bit_float, debug_path);


    for (int i = 0; i < size; ++i) {
        float val = image_8bit_float->data[i];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        // 显式截断：先转为int(截断小数部分)，再转为unsigned char
        image_8bit->data[i] = (unsigned char)(int)val;
    }
    sprintf(debug_path, "%s/c_10_final_8bit.txt", debug_folder);
    save_image_u8_to_txt(image_8bit, debug_path);

    // Free all intermediate images
    free_image_f(image_normalized);
    free_image_f(base_layer);
    free_image_f(detail_layer);
    free_image_f(base_layer_log);
    free_image_f(base_layer_compressed);
    free_image_f(fused_image);
    free_image_f(image_8bit_float);

    return image_8bit;
}

// --- Main Execution Logic ---

#ifdef _WIN32
void process_directory(const char *root_dir, const char *output_root_dir, const char *debug_dir) {
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

    printf("Found files to process...\\n");
    do {
        char raw_path[MAX_PATH];
        sprintf(raw_path, "%s\\%s", root_dir, find_data.cFileName);
        
        char stem[MAX_PATH];
        strcpy(stem, find_data.cFileName);
        char *dot = strrchr(stem, '.');
        if (dot) *dot = '\0';

        char image_debug_dir[MAX_PATH];
        sprintf(image_debug_dir, "%s\\%s", debug_dir, stem);
        CreateDirectoryA(image_debug_dir, NULL);

        printf("\\n--- Starting C Debug Run for %s ---\\n", find_data.cFileName);
        printf("Debug files will be saved in: %s\\n", image_debug_dir);

        ImageF *thermal_img_14bit = read_raw_thermal(raw_path, image_debug_dir);
        if (!thermal_img_14bit) continue;
        
        ImageU8 *final_image;
        float max_pix = 0.0f;
        for (int i = 0; i < thermal_img_14bit->width * thermal_img_14bit->height; ++i) {
            if (thermal_img_14bit->data[i] > max_pix) {
                max_pix = thermal_img_14bit->data[i];
            }
        }

        if (max_pix == 0) {
            final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
            printf("Info: File %s is all black, saving blank image.\\n", find_data.cFileName);
        } else {
            final_image = enhanced_thermal_conversion(
                thermal_img_14bit,
                guided_filter_radius,
                guided_filter_eps,
                base_min_percentile,
                base_max_percentile,
                base_layer_weight,
                detail_layer_weight,
                image_debug_dir
            );
        }
        
        char output_path[MAX_PATH];
        sprintf(output_path, "%s\\%s_debug.bmp", output_root_dir, stem);
        
        if(final_image) {
            write_bmp_grayscale(output_path, final_image);
            printf("Saved debug BMP to %s\\n", output_path);
        }
        
        free_image_f(thermal_img_14bit);
        free_image_u8(final_image);

    } while (FindNextFileA(h_find, &find_data) != 0);

    FindClose(h_find);
}
#else // POSIX implementation for directory traversal
void process_directory(const char *root_dir, const char *output_root_dir, const char *debug_dir) {
    DIR *d = opendir(root_dir);
    if (!d) {
        printf("Error: Input directory not found %s\n", root_dir);
        return;
    }

    // --- Adjustable Parameters ---
    int guided_filter_radius = 16;
    float guided_filter_eps = 0.01f * 0.01f;
    float base_min_percentile = 0.005f;
    float base_max_percentile = 99.4f;
    float base_layer_weight = 0.5f;
    float detail_layer_weight = 1.0f;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        const char *ext = strrchr(dir->d_name, '.');
        if (ext && strcmp(ext, ".raw") == 0) {
            char raw_path[1024];
            snprintf(raw_path, sizeof(raw_path), "%s/%s", root_dir, dir->d_name);
            
            char stem[256];
            strncpy(stem, dir->d_name, sizeof(stem));
            char *dot = strrchr(stem, '.');
            if (dot) *dot = '\0';

            char image_debug_dir[1024];
            snprintf(image_debug_dir, sizeof(image_debug_dir), "%s/%s", debug_dir, stem);
            mkdir(image_debug_dir, 0777);

            printf("\\n--- Starting C Debug Run for %s ---\\n", dir->d_name);
            printf("Debug files will be saved in: %s\\n", image_debug_dir);


            ImageF *thermal_img_14bit = read_raw_thermal(raw_path, image_debug_dir);
            if (!thermal_img_14bit) continue;
            
            ImageU8 *final_image;
            float max_pix = 0.0f;
            for (int i = 0; i < thermal_img_14bit->width * thermal_img_14bit->height; ++i) {
                if (thermal_img_14bit->data[i] > max_pix) {
                    max_pix = thermal_img_14bit->data[i];
                }
            }

            if (max_pix == 0) {
                final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
                printf("Info: File %s is all black, saving blank image.\\n", dir->d_name);
            } else {
                final_image = enhanced_thermal_conversion(
                    thermal_img_14bit,
                    guided_filter_radius,
                    guided_filter_eps,
                    base_min_percentile,
                    base_max_percentile,
                    base_layer_weight,
                    detail_layer_weight,
                    image_debug_dir
                );
            }
            
            char output_path[1024];
            snprintf(output_path, sizeof(output_path), "%s/%s_debug.bmp", output_root_dir, stem);

            if(final_image) {
                write_bmp_grayscale(output_path, final_image);
                printf("Saved debug BMP to %s\\n", output_path);
            }
            
            free_image_f(thermal_img_14bit);
            free_image_u8(final_image);
        }
    }
    closedir(d);
}
#endif

int main() {
    const char *root_dir = "sanyuan_data/14bit_test_image";
    const char *output_root_dir = "sanyuan_data/7_21_results_c";
    const char *debug_dir = "debug_output";

    #ifdef _WIN32
        CreateDirectoryA(output_root_dir, NULL);
        CreateDirectoryA(debug_dir, NULL);
    #else
        mkdir(output_root_dir, 0777);
        mkdir(debug_dir, 0777);
    #endif

    struct stat st;
    if (stat(root_dir, &st) != 0) {
        printf("Error: Input directory not found: %s\n", root_dir);
        return 1;
    }
    
    printf("--- Starting C Debug Run ---\n");
    process_directory(root_dir, output_root_dir, debug_dir);
    printf("\n--- C Debug Run Complete ---\n");

    return 0;
} 