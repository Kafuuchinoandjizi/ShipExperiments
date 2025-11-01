#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

// --- ARM NEON Intrinsics Header ---
#include <arm_neon.h>

// --- OpenMP Header for Parallelism ---
#include <omp.h>

// --- 定义常量 ---
#define WIDTH 640
#define HEIGHT 512
#define CHANNELS 1 // 灰度图
#define ALIGNMENT 64 // Cortex-A53 Cache Line Size
#define IMAGE_SIZE (WIDTH * HEIGHT)

// --- 图像结构体 (使用64字节对齐内存) ---
typedef struct {
    int W, H;
float *data;
} ImageF;

typedef struct {
    int W, H;
uint8_t *data;
} ImageU8;

// --- 全局内存分配和释放 (使用posix_memalign保证对齐) ---

void* aligned_malloc(size_t size) {
    void* ptr = NULL;
// 强制使用64字节对齐
if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
    perror("posix_memalign failed");
return NULL;
}
return ptr;
}

void aligned_free(void* ptr) {
free(ptr);
}

ImageF* create_image_f(int W, int H) {
ImageF *img = (ImageF*)malloc(sizeof(ImageF));
img->W = W;
img->H = H;
img->data = (float*)aligned_malloc(W * H * sizeof(float));
if (!img->data) {
free(img);
return NULL;
}
return img;
}

void free_image_f(ImageF *img) {
if (img) {
aligned_free(img->data);
free(img);
}
}

// 这是缺失的函数定义
void free_image_u8(ImageU8 *img) {
    if (img) {
        aligned_free(img->data);
        free(img);
    }
}


// --- Log查找表 (用于快速 log 计算，代替复杂的`logf`) ---
#define LOG_LUT_SIZE 256
float log_lut[LOG_LUT_SIZE];
float log_lut_scale;

void init_log_lut() {
// 假设输入float值范围在 [1.0, 256.0] 附近，进行查找表初始化
// OpenMP 并行初始化查找表
#pragma omp parallel for schedule(static)
for (int i = 0; i < LOG_LUT_SIZE; i++) {
    log_lut[i] = logf((float)i + 1.0f); // 避免 log(0)
}
log_lut_scale = 1.0f; // 简单查表，实际应用中可能需要更复杂的插值
}

// 快速查表 logf 近似 (NEON优化困难，但查表本身比`logf`快)
static inline float fast_logf(float x) {
if (x < 1.0f) x = 1.0f; // 避免负数或小于1的查表问题
if (x >= LOG_LUT_SIZE) return logf(x); // 超过范围使用标准logf
int idx = (int)x;
return log_lut[idx];
}


// --- I/O 和预处理 (NEON 优化: 14-bit转float) ---

                                               // NEON优化的14-bit RAW数据读取和转换
ImageF* read_raw_thermal_neon(const char *filename) {
FILE *fp = fopen(filename, "rb");
if (!fp) {
perror("Error opening file");
return NULL;
}

ImageF *img = create_image_f(WIDTH, HEIGHT);
if (!img) {
fclose(fp);
return NULL;
}

// 14-bit data is stored in a 16-bit array
uint16_t *raw_data = (uint16_t*)aligned_malloc(IMAGE_SIZE * sizeof(uint16_t));
if (!raw_data) {
free_image_f(img);
fclose(fp);
return NULL;
}

if (fread(raw_data, sizeof(uint16_t), IMAGE_SIZE, fp) != IMAGE_SIZE) {
fprintf(stderr, "Error reading raw data.\n");
aligned_free(raw_data);
free_image_f(img);
fclose(fp);
return NULL;
}
fclose(fp);

float *out = img->data;

// NEON 向量化转换
#pragma omp parallel for schedule(static)
for (int i = 0; i < IMAGE_SIZE; i += 4) {
    // Load 4x uint16_t values
    uint16x4_t v_u16 = vld1_u16(raw_data + i);

    // 提取 14 bit 数据 (假设数据在低14位)
    // 实际上14-bit数据可能跨字节存储或在16-bit内，这里按最常见的高位补0处理
    uint16x4_t v_data = vand_u16(v_u16, vdup_n_u16(0x3FFF));

    // 扩展到 32-bit (低位)
    uint32x4_t v_u32 = vmovl_u16(v_data);

    // 转换到 float32x4_t (NEON FPU)
    float32x4_t v_f32 = vcvtq_f32_u32(v_u32);

    // 存储结果
    vst1q_f32(out + i, v_f32);
}

aligned_free(raw_data);
return img;
}

// --- O(1) 积分图像 Box Filter ---

                         // NEON优化的积分图像生成 (水平累加向量化)
ImageF* integral_image_neon(const ImageF *src) {
ImageF *ii = create_image_f(src->W, src->H);
if (!ii) return NULL;

const float *src_data = src->data;
float *ii_data = ii->data;
int W = src->W;
int H = src->H;

// Phase 1: 水平累加 (并行化行)
#pragma omp parallel for schedule(static)
for (int y = 0; y < H; y++) {
    float *ii_row = ii_data + y * W;
const float *src_row = src_data + y * W;

// 首元素
ii_row[0] = src_row[0];

// 向量化累加 (x > 0)
for (int x = 4; x < W; x += 4) {
float32x4_t v_src = vld1q_f32(src_row + x);
float32x4_t v_prev = vld1q_f32(ii_row + x - 4);

// 累加操作的串行性需要特殊处理，但简单累加可以这样进行：
// v_sum[0] = src[x] + ii[x-1]
// v_sum[1] = src[x+1] + ii[x]
// ...
// 简单地 v_prev + v_src 是错误的。为了确保正确性，这里采用混合方法：
// 1. NEON 向量化行内累加 (可能牺牲部分效率但保证正确性)
float accum = src_row[x-1];
for(int k=x; k<x+4 && k<W; ++k){
accum += src_row[k];
ii_row[k] = accum;
}
}
// 边界处理
for(int x=1; x<W; ++x) {
ii_row[x] = src_row[x] + ii_row[x-1];
}
}

// Phase 2: 垂直累加 (串行y，但仍然OpenMP并行化行)
#pragma omp parallel for schedule(static)
for (int y = 1; y < H; y++) {
    float *ii_curr = ii_data + y * W;
float *ii_prev = ii_data + (y - 1) * W;

// 垂直累加，可NEON向量化
for (int x = 0; x < W; x += 4) {
if (x + 4 > W) { // 边界处理
for (int k = x; k < W; k++) {
ii_curr[k] += ii_prev[k];
}
break;
}
float32x4_t v_curr = vld1q_f32(ii_curr + x);
float32x4_t v_prev = vld1q_f32(ii_prev + x);

float32x4_t v_sum = vaddq_f32(v_curr, v_prev);

vst1q_f32(ii_curr + x, v_sum);
}
}

return ii;
}

// O(1) Box Filter Mean (使用积分图像)
ImageF* box_filter_mean_O1(const ImageF *II, int r) {
int W = II->W;
int H = II->H;
ImageF *mean_img = create_image_f(W, H);
if (!mean_img) return NULL;

float *mean_data = mean_img->data;
const float *ii_data = II->data;

const int R = r;
const float inv_area = 1.0f / ((2 * R + 1) * (2 * R + 1));

// OpenMP 并行计算均值
#pragma omp parallel for schedule(static)
for (int y = 0; y < H; y++) {
for (int x = 0; x < W; x++) {
// 计算边界
int x1 = x - R;
int y1 = y - R;
int x2 = x + R;
int y2 = y + R;

// 裁剪边界
x1 = x1 < 0 ? 0 : x1;
y1 = y1 < 0 ? 0 : y1;
x2 = x2 >= W ? W - 1 : x2;
y2 = y2 >= H ? H - 1 : y2;

// 积分图像 O(1) 查找
float A = (x1 > 0 && y1 > 0) ? ii_data[y1 * W + x1] : 0.0f;
float B = (y1 > 0) ? ii_data[y1 * W + x2] : 0.0f;
float C = (x1 > 0) ? ii_data[y2 * W + x1] : 0.0f;
float D = ii_data[y2 * W + x2];

float sum = D - B - C + A;

// 实际窗口面积
float actual_area = (float)(x2 - x1 + 1) * (y2 - y1 + 1);

mean_data[y * W + x] = sum / actual_area;
}
}

return mean_img;
}

// --- NEON 优化的导向滤波核心函数 ---

ImageF* guided_filter_neon(const ImageF *I, const ImageF *P, int r, float eps) {
int W = I->W;
int H = I->H;
int N = W * H;

// 1. 预计算 I^2 和 I*P
ImageF *I_sq = create_image_f(W, H);
ImageF *I_P = create_image_f(W, H);

// NEON 向量化计算 I^2 和 I*P
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i += 4) {
if (i + 4 > N) break;
float32x4_t v_I = vld1q_f32(I->data + i);
float32x4_t v_P = vld1q_f32(P->data + i);

float32x4_t v_I_sq = vmulq_f32(v_I, v_I);
float32x4_t v_I_P = vmulq_f32(v_I, v_P);

vst1q_f32(I_sq->data + i, v_I_sq);
vst1q_f32(I_P->data + i, v_I_P);
}

// 2. 计算积分图像 (Mean)
ImageF *II_I = integral_image_neon(I);
ImageF *II_P = integral_image_neon(P);
ImageF *II_I_sq = integral_image_neon(I_sq);
ImageF *II_I_P = integral_image_neon(I_P);

// 3. 计算局部均值 (Box Filter O(1) 查找)
ImageF *mean_I = box_filter_mean_O1(II_I, r);
ImageF *mean_P = box_filter_mean_O1(II_P, r);
ImageF *mean_I_sq = box_filter_mean_O1(II_I_sq, r);
ImageF *mean_I_P = box_filter_mean_O1(II_I_P, r);

// 4. 计算局部协方差和方差
ImageF *cov_I_P = create_image_f(W, H); // cov(I, P)
ImageF *var_I = create_image_f(W, H);   // var(I)

                                           // NEON 向量化计算 cov 和 var
                                                                     // cov(I, P) = mean(I*P) - mean(I) * mean(P)
                                                                                    // var(I) = mean(I^2) - mean(I)^2
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i += 4) {
    if (i + 4 > N) break;

    float32x4_t v_mean_I = vld1q_f32(mean_I->data + i);
    float32x4_t v_mean_P = vld1q_f32(mean_P->data + i);
    float32x4_t v_mean_I_sq = vld1q_f32(mean_I_sq->data + i);
    float32x4_t v_mean_I_P = vld1q_f32(mean_I_P->data + i);

    // mean_I * mean_P
    float32x4_t v_I_mul_P = vmulq_f32(v_mean_I, v_mean_P);
    // mean_I * mean_I
    float32x4_t v_I_sq_mean = vmulq_f32(v_mean_I, v_mean_I);

    // cov_I_P = mean_I_P - mean_I * mean_P
    float32x4_t v_cov = vsubq_f32(v_mean_I_P, v_I_mul_P);

    // var_I = mean_I_sq - mean_I^2
    float32x4_t v_var = vsubq_f32(v_mean_I_sq, v_I_sq_mean);

    vst1q_f32(cov_I_P->data + i, v_cov);
    vst1q_f32(var_I->data + i, v_var);
    }

    // 5. 计算线性系数 a 和 b
    ImageF *A_map = create_image_f(W, H);
    ImageF *B_map = create_image_f(W, H);
    float32x4_t v_eps = vdupq_n_f32(eps);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    // NEON 向量化计算 a 和 b
                            // a = cov / (var + eps)
                                   // b = mean_P - a * mean_I
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += 4) {
        if (i + 4 > N) break;

        float32x4_t v_cov = vld1q_f32(cov_I_P->data + i);
        float32x4_t v_var = vld1q_f32(var_I->data + i);
        float32x4_t v_mean_I = vld1q_f32(mean_I->data + i);
        float32x4_t v_mean_P = vld1q_f32(mean_P->data + i);

        // 分母: var + eps
        float32x4_t v_denom = vaddq_f32(v_var, v_eps);

        // ARM NEON 没有直接的向量除法，使用倒数近似 (Reciprocal Estimate)
        // a = cov * (1 / denom)
        // VRECPE: Reciprocal Estimate. VRECPS: Reciprocal Step. Combined for better accuracy.
            float32x4_t v_rcp = vrecpeq_f32(v_denom);
        v_rcp = vmulq_f32(vrecpsq_f32(v_denom, v_rcp), v_rcp);

        float32x4_t v_a = vmulq_f32(v_cov, v_rcp);

        // b = mean_P - a * mean_I (使用 vmlsq_f32 - Multiply-Subtract)
        float32x4_t v_b = vmlsq_f32(v_mean_P, v_a, v_mean_I);

        vst1q_f32(A_map->data + i, v_a);
        vst1q_f32(B_map->data + i, v_b);
        }

        // 6. 均值化 a 和 b
        ImageF *II_A = integral_image_neon(A_map);
        ImageF *II_B = integral_image_neon(B_map);

        ImageF *mean_A = box_filter_mean_O1(II_A, r);
        ImageF *mean_B = box_filter_mean_O1(II_B, r);

        // 7. 最终线性融合
        ImageF *Q = create_image_f(W, H);

        // NEON 向量化最终融合
                // Q = mean_A * I + mean_B
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i += 4) {
            if (i + 4 > N) break;

            float32x4_t v_I = vld1q_f32(I->data + i);
            float32x4_t v_mean_A = vld1q_f32(mean_A->data + i);
            float32x4_t v_mean_B = vld1q_f32(mean_B->data + i);

            // vmlaq_f32: Multiply-Accumulate: v_sum = v_mean_B + v_mean_A * v_I
            float32x4_t v_Q = vmlaq_f32(v_mean_B, v_mean_A, v_I);

            vst1q_f32(Q->data + i, v_Q);
            }

            // --- 释放中间内存 --- \
            free_image_f(I_sq); free_image_f(I_P);
            free_image_f(II_I); free_image_f(II_P); free_image_f(II_I_sq); free_image_f(II_I_P);
            free_image_f(mean_I); free_image_f(mean_P); free_image_f(mean_I_sq); free_image_f(mean_I_P);
            free_image_f(cov_I_P); free_image_f(var_I);
            free_image_f(A_map); free_image_f(B_map);
            free_image_f(II_A); free_image_f(II_B);
            free_image_f(mean_A); free_image_f(mean_B);

            return Q;
        }

        // --- 后处理和保存 (NEON 优化: float转uint8_t并饱和) ---

// --- 后处理和保存 (NEON 优化: float转uint8_t并饱和) ---

ImageU8* normalize_and_convert_neon(const ImageF *src) {
    ImageU8 *dst = (ImageU8*)malloc(sizeof(ImageU8));
    dst->W = src->W;
    dst->H = src->H;
    dst->data = (uint8_t*)aligned_malloc(src->W * src->H * sizeof(uint8_t));
    if (!dst->data) {
        free(dst);
        return NULL;
    }

    const float *src_data = src->data;
    uint8_t *dst_data = dst->data;
    int N = src->W * src->H;

    // 找到全局最小值和最大值
    float min_val = src_data[0];
    float max_val = src_data[0];
    // 使用 NEON 向量化查找 Min/Max 可以进一步加速，此处为简洁仍使用串行
    for(int i = 1; i < N; ++i) {
        if (src_data[i] < min_val) min_val = src_data[i];
        if (src_data[i] > max_val) max_val = src_data[i];
    }

    // 归一化参数
    float range = max_val - min_val;
    float scale = (range > 1e-6) ? 255.0f / range : 0.0f;

    float32x4_t v_min = vdupq_n_f32(min_val);
    float32x4_t v_scale = vdupq_n_f32(scale);

    // NEON 向量化归一化和转换
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += 4) {
        if (i + 4 > N) break;

        float32x4_t v_src = vld1q_f32(src_data + i);

        // 1. (v_src - v_min) * v_scale
        float32x4_t v_temp = vsubq_f32(v_src, v_min);
        float32x4_t v_norm = vmulq_f32(v_temp, v_scale);

        // 2. 转换为 int32 (四舍五入: +0.5)
        float32x4_t v_round = vaddq_f32(v_norm, vdupq_n_f32(0.5f));
        int32x4_t v_i32 = vcvtq_s32_f32(v_round); // 饱和到int32

        // 3. 饱和压缩到 uint8
        uint16x4_t v_u16 = vqmovun_s32(v_i32); // 饱和压缩 S32 -> U16 (4x U16)

        // 组合成 128-bit 向量以使用 vqmovn_u16 进行饱和压缩
        uint8x8_t v_u8_combined = vqmovn_u16(vcombine_u16(v_u16, v_u16)); // 饱和压缩 U16[8] -> U8[8]

        // 4. 存储结果 (只存低4个/32位)
        // 核心修正：使用 vget_low_u8() 期望 128-bit 向量，但 v_u8_combined 是 64-bit。
        // 正确做法是存储 v_u8_combined 的低 32 位，即 4 字节。
        uint32x2_t v_u32_2 = vreinterpret_u32_u8(v_u8_combined);
        vst1_lane_u32((uint32_t*)(dst_data + i), v_u32_2, 0);
    }

    return dst;
}

// 简化版 BMP 写入 (仅支持 8-bit 灰度图)
void write_bmp(const char *filename, const ImageU8 *img) {
// [省略 BMP 头文件写入代码，假设外部工具有更可靠的实现]
// 关键是写入实际像素数据 (img->data)
// 为简洁和可移植性，这里只打印保存信息
printf("--- [Image Save Placeholder] Saving 8-bit image to %s\n", filename);
// 实际项目中应包含完整的文件头写入逻辑
}

// --- Main Loop ---

void process_directory(const char *root_dir, const char *output_root_dir, int r, float eps) {
DIR *d;
struct dirent *dir;
d = opendir(root_dir);
if (!d) {
fprintf(stderr, "Cannot open directory %s\n", root_dir);
return;
}

// 设置OpenMP线程数
omp_set_num_threads(4);
printf("--- OpenMP Threads: 4 cores ---\n");

while ((dir = readdir(d)) != NULL) {
if (strstr(dir->d_name, ".raw") != NULL) {
char input_path[512], output_path[512], stem[512];
snprintf(input_path, sizeof(input_path), "%s/%s", root_dir, dir->d_name);

// 提取文件名主干
char *dot = strrchr(dir->d_name, '.');
size_t len = dot ? (size_t)(dot - dir->d_name) : strlen(dir->d_name);
strncpy(stem, dir->d_name, len);
stem[len] = '\0';
snprintf(output_path, sizeof(output_path), "%s/%s_opt.bmp", output_root_dir, stem);

printf("Processing: %s\n", dir->d_name);

// --- 计时开始 ---
double start_time = omp_get_wtime();

// 1. NEON 优化读取和转换 (14bit -> float)
ImageF *thermal_img_14bit = read_raw_thermal_neon(input_path);
if (!thermal_img_14bit) continue;

// 2. 核心算法: NEON & O(1) 导向滤波
// P (Input) = thermal_img_14bit
// I (Guidance) = thermal_img_14bit (Self-guided filter)
ImageF *filtered_image = guided_filter_neon(
    thermal_img_14bit,
    thermal_img_14bit,
    r,
    eps
);

// 3. NEON 优化归一化和转换 (float -> uint8_t)
ImageU8 *final_image = normalize_and_convert_neon(filtered_image);

// 4. 保存文件 (简略版)
write_bmp(output_path, final_image);

// --- 计时结束 ---
double processing_time = omp_get_wtime() - start_time;

printf("Processed: %s -> %s_opt.bmp (Wall-Clock Time: %.4f s)\n\n", dir->d_name, stem, processing_time);

free_image_f(thermal_img_14bit);
free_image_f(filtered_image);
free_image_u8(final_image);
}
}
closedir(d);
}

int main() {
           // 假设路径
const char *root_dir = "./RAW_DATA/"; // 请修改为您的输入路径
const char *output_root_dir = "./test_results_arm_optimized"; // 请修改为您的输出路径

                                                                 // 导向滤波参数
const int radius = 7;
const float regularization_eps = 0.0001f;

// 初始化对数查找表
init_log_lut();

// 创建输出目录
mkdir(output_root_dir, 0777);

printf("=== ARM Cortex-A53 Aggressively Optimized Thermal Image Conversion ===\n");
printf("--- Optimization Strategy: O(1) Integral Image + NEON SIMD ---\n");
printf("--- Filter Parameters: R=%d, Epsilon=%.4f ---\n", radius, regularization_eps);

struct stat st;
if (stat(root_dir, &st) != 0) {
    printf("Error: Input directory not found: %s. Please create this directory and place .raw files in it.\n", root_dir);
// 为了演示，不退出，让用户知道需要创建目录
} else {
    process_directory(root_dir, output_root_dir, radius, regularization_eps);
}

return 0;
}