#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <c6x.h>
#include <ti/csl/csl.h>
#include <ti/csl/csl_chip.h>
#include <ti/csl/csl_cache.h>
#include <ti/csl/csl_cacheAux.h>
#include <ti/csl/csl_sem.h>
#include <ti/csl/csl_semAux.h>
//#include "common.h"
//#include "sense_net.h"
#include "commondef.h"
//#include "datatrans.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mathlib.h>
#include "c6x.h"

extern "C"
{
#include <dsplib.h>
}



unsigned int g_startDDR3Address = USER_DDR_ADDR;//0x8A000000
unsigned int g_startMSMCAddress = USER_MSMC_ADDR;
unsigned int g_startL2Address = USER_HEAP_ADDR;





inline float com_div3(float a, float b){
    float rcpsp_temp1 = _rcpsp(b); // 初始近似值
    float rcpsp_temp2 = rcpsp_temp1 * (2 - rcpsp_temp1 * b);
    float rcpsp_temp3 = rcpsp_temp2 * (2 - rcpsp_temp2 * b);
    return a * rcpsp_temp3;
}

inline float com_div2(float a, float b){
    float rcpsp_temp1 = _rcpsp(b); // 初始近似值
    float rcpsp_temp2 = rcpsp_temp1 * (2 - rcpsp_temp1 * b);
    return a * rcpsp_temp2;
}
inline float com_div1(float a, float b){
    float rcpsp_temp1 = _rcpsp(b); // 初始近似值
    return a * rcpsp_temp1;
}
/******************************************
* *   Function :matrix_min
-------------------------------------------
* *   Brief    :求图像数据的最小值
-------------------------------------------
* *   Param    : input 输入图像数据指针
                    height 图像高度
                    width  图像宽度
-------------------------------------------
* *   Retval   : null
-------------------------------------------
* *   Note     : null
*******************************************/
inline float matrix_min(void* input, const int height, const int width)
{
    int i;
    const int len = height * width / 2;

    unsigned int* p1 = (unsigned int*)input;
    unsigned int x1, x2, x3, x4;
    long long templ1, templ2, templ3, templ4;

    templ1 = _itoll(p1[0], p1[1]);
    templ2 = _itoll(p1[2], p1[3]);
    templ3 = _dmin2(templ1, templ2);
    for (i = 4; i < len; i += 4) {
        templ1 = _itoll(p1[i], p1[i + 1]);
        templ2 = _itoll(p1[i + 2], p1[i + 3]);
        templ4 = _dmin2(templ1, templ2);
        templ3 = _dmin2(templ3, templ4);
    }
    x1 = templ3 >> 48 & 0xffff;
    x2 = templ3 >> 32 & 0xffff;
    x3 = templ3 >> 16 & 0xffff;
    x4 = templ3 & 0xffff;
    x1 = x1 > x2 ? x2 : x1;
    x3 = x3 > x4 ? x4 : x3;
    return x1 > x3 ? x3 : x1;
}


/******************************************
* *   Function :convert_signed_char_float
-------------------------------------------
* *   Brief    :将16位整数转化为浮点数
-------------------------------------------
* *   Param    : input 输入的数据指针地址
                    output 输出的浮点数地址
                    height 图像高度
                    width 图像宽度
-------------------------------------------
* *   Retval   : null
-------------------------------------------
* *   Note     : null
*******************************************/
inline void convert_signed_char_float( void* input, void* output, const int height, const int width)
{
    int i;
    const int max_value = 0x3FFF3FFF;
//     long long* p1 = input;
    unsigned int* p1 = (unsigned int*)input;
    const int len = height * width * 0.5;

    for (i = 0; i < len; i++) {
        //   p1[i] = _dmin2(p1[i],max_value);
        p1[i] &= max_value;
    }

//    unsigned int* p2 = input;
    __float2_t* restrict p2 = (__float2_t*)output;
//    len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        p2[i] = _dinthspu(p1[i]);
    }
}


//inline void convert_float_char(const float* input, void* output, const int height, const int width)
//{
//    int i;
//    const float* restrict p1 = input;
//    unsigned char* restrict p2 = (unsigned char*)output;
//    long long* restrict p3 = (long long*)output;
//
//    const long long temp1 = _itoll(0xFF, 0xFF);  // 正确的裁剪范围
//    int len = height * width;
//
//    // float -> uint8_t (via _spint)
//    for (i = 0; i < len; i++) {
//        p2[i] = _spint(p1[i]);
//    }
//
//    // Clamp to [0, 255] for every 4 bytes
//    len = len / 4;
//    for (i = 0; i < len; i++) {
//        p3[i] = _dminu4(p3[i], temp1);
//    }
//}
inline void convert_float_char(const float* input, void* output, const int height, const int width)
{
    int i;
    const float* restrict p1 = input;
    unsigned char* restrict p2 = (unsigned char*)output;
//    const int max_value = 0xFAFAFAFA;
    const int max_value = 0xFFFFFFFF;
//    long long* restrict p3 = (long long*)output;
    const long long temp1 = _itoll(max_value, max_value);
    int len = height * width;
    for (i = 0; i < len; i++) {
        p2[i] = _spint(p1[i]);
    }

//    len = len / 4;
//    for (i = 0; i < len; i++) {
//        p3[i] = _dminu4(p3[i], temp1);
//    }
}
inline void matrix_mul(void* input, void* output,const float beta, int height, int width)
{
    int i;
    __float2_t* restrict p1 = (__float2_t*)input;
    __float2_t* restrict p2 = (__float2_t*)output;
    __float2_t temp0 = _ftof2(beta, beta);

    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        p2[i] = _dmpysp(temp0, p1[i]);
    }
}

//inline void matrixxxxxx(void* input, void* output, const float beta,const float beta1, int height, int width)
//{
//    int i;
//    __float2_t* restrict p1 = (__float2_t*)input;
//    __float2_t* restrict p2 = (__float2_t*)output;
//    __float2_t temp0;
//    __float2_t temp1;
//    __float2_t temp2;
//    temp0 = _ftof2(beta, beta);
//    temp1 = _ftof2(beta1, beta1);
//    const int len = height * width * 0.5;
//    for (i = 0; i < len; i++) {
//    	temp2 = _dsub2(p1[i],temp0);
//        p2[i] = _dmpysp(temp2, temp1);
//    }
//}
inline void matrix_sub(void* input1, void* input2, void* output, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)output;
    int i;

    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        p3[i] = _dsubsp(p1[i], p2[i]);
    }
}

inline void loglog(void* input1, void* output, float beta, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)output;
    int i;

    const __float2_t temp0 = _ftof2(1.0, 1.0);
    const __float2_t temp1 = _ftof2(beta, beta);
    __float2_t temp2;
    __float2_t temp3;

    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp2= _daddsp(p1[i],temp0);

        float val_low = _lof2(temp2);
        float val_high = _hif2(temp2);

        // Compute logarithms
        float res_low = logsp(val_low);
        float res_high = logsp(val_high);

        temp3 = _ftof2(res_low, res_high);

        p2[i] = _dmpysp(temp1, temp3);

    }


}

//inline void matrix_add(void* input1, void* input2, void* output, float beta, int height, int width)
//{
//    __float2_t* restrict p1 = input1;
//    __float2_t* restrict p2 = input2;
//    __float2_t* restrict p3 = output;
//    int i;
//    const __float2_t temp0 = _ftof2(beta, beta);
//    __float2_t temp1;
//
//    const int len = height * width * 0.5;
//    for (i = 0; i < len; i++) {
//        temp1 = _dmpysp(p2[i], temp0); //input2[i] * beta;
//        p3[i] = _daddsp(p1[i], temp1); //input1[i] + input2[i] * beta;
//    }
//}
float matrix_max(void* input, int height, int width)
{
    int i;
    const int len = height * width / 2;

    unsigned int* p1 = (unsigned int*)input;
    unsigned int x1, x2, x3, x4;
    long long templ1, templ2, templ3, templ4;

    templ1 = _itoll(p1[0], p1[1]);
    templ2 = _itoll(p1[2], p1[3]);
    templ3 = _dmin2(templ1, templ2);
    for (i = 4; i < len; i += 4) {
        templ1 = _itoll(p1[i], p1[i + 1]);
        templ2 = _itoll(p1[i + 2], p1[i + 3]);
        templ4 = _dmax2(templ1, templ2);
        templ3 = _dmax2(templ3, templ4);
    }
    x1 = templ3 >> 48 & 0xffff;
    x2 = templ3 >> 32 & 0xffff;
    x3 = templ3 >> 16 & 0xffff;
    x4 = templ3 & 0xffff;
    x1 = x1 < x2 ? x2 : x1;
    x3 = x3, x4 ? x4 : x3;
    return x1 < x3 ? x3 : x1;

}
void set_cache(uint32_t DDR_start_addr, uint32_t DDR_end_addr)
{
    uint32_t MAR = 0;

    while (DDR_start_addr <= DDR_end_addr)
    {
        // DDR 起始地址 ：0x80000000
        if (DDR_start_addr >= 0x80000000 && DDR_start_addr <= 0xFFFFFFFF)
        {
            MAR = (DDR_start_addr - 0x80000000) / 0xFFFFFF + 128;
            CACHE_enableCaching(MAR);
            // CACHE_setMemRegionInfo(MAR, 1, 1);
            DDR_start_addr += 0xFFFFFF;
        }
    }
}
inline void matrix_matrix_mul(void* input1, void* input2, void* output, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)output;
    int i;

    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        p3[i] = _dmpysp(p1[i], p2[i]);
    }
}

inline void matrix_matrix_mul_sub(void* input1, void* input2, void* input3, void* output, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)input3;
    __float2_t* restrict p4 = (__float2_t*)output;
    int i;
    __float2_t temp;
    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp = _dmpysp(p2[i], p3[i]);
        p4[i]= _dsubsp(p1[i],temp);
    }
}

inline void matrix_matrix_sub_mul(void* input1, void* input2, void* output, float beta, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)output;
    int i;
    __float2_t temp0 = _ftof2(beta, beta);

    __float2_t temp;
    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp= _dsubsp(temp0,p1[i]);
        p3[i] = _dmpysp(temp, p2[i]);
    }
}

inline void matrix_sub_mul(void* input1, void* output, float beta, float beta1, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)output;
    int i;
    const __float2_t temp0 = _ftof2(beta, beta);
    const __float2_t temp1 = _ftof2(beta1, beta1);

    __float2_t temp;
    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp = _dsubsp(p1[i],temp0);
        p2[i] = _dmpysp(temp1, temp);
    }
}

inline void matrix_mul_add(void* input1, void* input2, void* input3, void* output, int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)input3;
    __float2_t* restrict p4 = (__float2_t*)output;
    int i;
    __float2_t temp;
    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp = _dmpysp(p2[i], p3[i]);
        p4[i]= _daddsp(p1[i],temp);
    }
}
inline void matrix_add(void* input1, void* input2, void* output, float beta, float beta1,int height, int width)
{
    __float2_t* restrict p1 = (__float2_t*)input1;
    __float2_t* restrict p2 = (__float2_t*)input2;
    __float2_t* restrict p3 = (__float2_t*)output;
    int i;
    const __float2_t temp0 = _ftof2(beta, beta);
    const __float2_t temp1 = _ftof2(beta1, beta1);
    __float2_t temp2;
    __float2_t temp3;

    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        temp2 = _dmpysp(p1[i], temp0); //input1[i] * beta;
        temp3 = _dmpysp(p2[i], temp1); //input2[i] * beta1;
        p3[i] = _daddsp(temp2, temp3); //input1[i]* beta0 + input2[i] * beta1;
    }
}


int partition(float *arr, int left, int right, int pivotIndex) {
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
float quick_select(float *arr, int left, int right, int k) {
    if (left == right) return arr[left];

    int pivotIndex = left + rand() % (right - left + 1);
    pivotIndex = partition(arr, left, right, pivotIndex);

    if (k == pivotIndex){
        return arr[k];
    } else if (k < pivotIndex){
        return quick_select(arr, left, pivotIndex - 1, k);
    } else {
        return quick_select(arr, pivotIndex + 1, right, k);
    }
}



void* snet_malloc(int i_size, unsigned int* startAddress)
{
    unsigned char *align_buf = NULL;
    unsigned int curAddr = *startAddress;
    curAddr = (curAddr + 63) & (0xFFFFFFC0);
    align_buf = (unsigned char*)curAddr;
    curAddr += i_size;
    *startAddress = curAddr;
    //printf("L2:%d, MSM:%d, DDR3:%d\n", g_startHeapAddress-HEAP_STRUCTURE_ADDR, g_startMSMCAddress-MSMC_STRUCTURE_ADDR, g_startDDR3Address-DDR3_STRUCTURE_ADDR);
    return (void*)align_buf;
}


#include <float.h>  // 添加这个头文件以使用DBL_MAX



#define M_LN2 0.693147180559945309417


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





void save_image_u16_to_txt(const uint16_t* img, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open file for writing");
        return;
    }
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 320; ++x) {
            fprintf(f, "%d ", img[y * 320 + x]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void save_image_u8_to_txt(const ImageU8* img, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("DSP_ERROR: Failed to open file '%s' for writing.\n", filename);
        return;
    }
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 320; ++x) {
            fprintf(f, "%d ", img->data[y * 320 + x]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("DSP_INFO: Saved u8 image to %s\n", filename);
}

void save_image_f_to_txt(const ImageF* img, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open file for writing");
        return;
    }
//    for (int y = 0; y < img->height; ++y) {
    for (int y = 0; y < img->height; ++y) {
        for (int x = 0; x < img->width; ++x) {
            fprintf(f, "%.8f ", img->data[y * img->width + x]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}
// --- Memory Management for Image Structs ---



ImageF* create_image_f(int width, int height) {
    ImageF *img = (ImageF*)malloc(sizeof(ImageF));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
//    img->data = (float*)malloc(width * height * sizeof(float));
    img->data = (float*)snet_malloc(width * height * sizeof(float), &g_startDDR3Address);

    return img;
}

ImageU8* create_image_u8(int width, int height) {
    ImageU8 *img = (ImageU8*)malloc(sizeof(ImageU8));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
//    img->data = (unsigned char*)calloc((size_t)width * height, sizeof(unsigned char));
//    unsigned int hhhh = USER_DDR_ADDR_test11;
//    img->data = (unsigned char*)snet_malloc(width * height * sizeof(unsigned char*), &hhhh);
//    g_startL2Address = USER_HEAP_ADDR;
    img->data = (unsigned char*)snet_malloc(width * height * sizeof(unsigned char), &g_startL2Address);

    return img;
}

// --- File I/O ---

long get_file_size(const char *filename) {
//    struct stat st;
//    if (stat(filename, &st) == 0) {
//        return st.st_size;
//    }
    return -1;
}

ImageF* read_raw_thermal(const char *fname) {

    int w = 320, h = 256;
    uint16_t *raw_data = (uint16_t*)snet_malloc(w * h * sizeof(uint16_t), &g_startL2Address);


    ImageF *img_out = create_image_f(w, h);

//    uint16_t *temp_row = (uint16_t *)malloc(w * sizeof(uint16_t));
    uint16_t *temp_row = (uint16_t *)snet_malloc(w * sizeof(uint16_t), &g_startL2Address);
    g_startL2Address = USER_HEAP_ADDR;
    if(temp_row) {
        memcpy(temp_row, raw_data + w, w * sizeof(uint16_t));
        memcpy(raw_data, temp_row, w * sizeof(uint16_t));
    }

//    int n_pixels_in_frame = w * h;
//    for (int i = 0; i < n_pixels_in_frame; ++i) {
//        uint16_t val = raw_data[i];
////            val = (val >> 8) | (val << 8);
//        img_out->data[i] = (float)(val & 0x3FFF);
//    }
    convert_signed_char_float( (void*)raw_data, (void*)img_out->data, h, w);

//    save_image_f_to_txt(img_out, "../Images/output.txt");
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
    float *sorted_data = (float*)snet_malloc(size * sizeof(float), &g_startL2Address);
    g_startL2Address = USER_HEAP_ADDR;
    memcpy(sorted_data, img->data, size * sizeof(float));
//    qsort(sorted_data, size, sizeof(float), compare_floats);
//
//
////    int index = (int)(((size_t)size - 1) * p / 100.0);
//        int index = (int)(((size_t)size - 1) * p * 0.01);
//    float result = sorted_data[index];

    int index = (int)(((float)(size - 1)) * p * 0.01);  // 索引位置
    float result = quick_select(sorted_data, 0, size - 1, index);
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

    ImageF* temp = create_image_f(w, h);

    // Vertical pass from src to temp, using float for accumulator
    for (int x = 0; x < w; ++x) {
        float sum = 0.0;
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

    // Horizontal pass from temp to out, using float for accumulator
    float norm = 1.0f / (float)((2 * r + 1) * (2 * r + 1));
    for (int y = 0; y < h; ++y) {
        float sum = 0.0;
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

    return out;
}






ImageF* guided_filter(const ImageF* src, int radius, float eps) {
    int w = src->width;
    int h = src->height;
    int size = w * h;

    const ImageF *I = src;

    ImageF* mean_I = box_filter(I, radius);
    if (!mean_I) return NULL;

    ImageF* I_sq = create_image_f(w, h);

//    for(int i=0; i<size; ++i) I_sq->data[i] = I->data[i] * I->data[i];
    matrix_matrix_mul(I->data, I->data, I_sq->data, h, w);

    ImageF* mean_I_sq = box_filter(I_sq, radius);

    ImageF* var_I = create_image_f(w, h);
//    for(int i=0; i<size; ++i) var_I->data[i] = mean_I_sq->data[i] - mean_I->data[i] * mean_I->data[i];
    matrix_matrix_mul_sub(mean_I_sq->data, mean_I->data, mean_I->data, var_I->data, h, w);

    ImageF* a = create_image_f(w, h);

    for(int i=0; i<size; ++i) a->data[i] = var_I->data[i] / (var_I->data[i] + eps);
//    for(int i=0; i<size; ++i) a->data[i] = com_div1(var_I->data[i] , (var_I->data[i] + eps));

    ImageF* b = create_image_f(w, h);

//    for(int i=0; i<size; ++i) b->data[i] = (1.0f - a->data[i]) * mean_I->data[i];
    matrix_matrix_sub_mul(a->data, mean_I->data, b->data, 1.0f, h, w);

    ImageF* mean_a = box_filter(a, radius);
    ImageF* mean_b = box_filter(b, radius);

    ImageF* q = create_image_f(w, h);
//    for(int i=0; i<size; ++i) q->data[i] = mean_a->data[i] * I->data[i] + mean_b->data[i];
    matrix_mul_add(mean_b->data, mean_a->data, I->data, q->data, h, w);

    return q;
}
inline void matrix_multest(void* input, const float beta, int height, int width)
{
    int i;
    __float2_t* restrict p1 = (__float2_t*)input;
    __float2_t temp0;
    temp0 = _ftof2(beta, beta);
    const int len = height * width * 0.5;
    for (i = 0; i < len; i++) {
        p1[i] = _dmpysp(temp0, p1[i]);
    }
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
    long long StartTime = 0;
    long long EndTime = 0;

    int w = image_14bit->width;
    int h = image_14bit->height;
    int size = w * h;
#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif
    // 1. Normalize to [0, 1]
    float temp_num = 1 / 16383.0f;
    ImageF *image_normalized = create_image_f(w, h);
//    for (int i = 0; i < size; ++i) {
////      image_normalized->data[i] = image_14bit->data[i] / 16383.0f; // 2^14 - 1
//        image_normalized->data[i] = image_14bit->data[i] * temp_num; // 2^14 - 1
//    }
//    matrix_mul((void*) image_14bit->data, (void*)image_normalized->data, temp_num, h, w);
    matrix_multest((void*) image_14bit->data, temp_num, h, w);
#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("1. Normalize to [0, 1] time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif


#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif
    // 2. Guided filter
    ImageF *base_layer = guided_filter(image_14bit, radius, eps);
    ImageF *detail_layer = create_image_f(w, h);
//#if test_time
//    StartTime = 0;
//    EndTime = 0;
//
//    TSCH = 0;
//    TSCL = 0;
//    StartTime = _itoll(TSCH, TSCL);
//#endif
//    for (int i = 0; i < size; ++i) {
//        detail_layer->data[i] = image_normalized->data[i] - base_layer->data[i];
//    }

    matrix_sub(image_14bit->data, base_layer->data, detail_layer->data, h, w);
#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("2. Guided filter time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif

#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif
    // 3. Log map on base layer
    ImageF *base_layer_log = create_image_f(w, h);


    float c = 1.0f / logsp(2.0f); // 1 / ln(2)
    for (int i = 0; i < size; ++i) {
        base_layer_log->data[i] = c * logsp(1.0f + base_layer->data[i]);
//    	 base_layer_log->data[i] = c * powsp(base_layer->data[i], (float)0.3);
//    	base_layer_log->data[i] = c * sqrtsp(base_layer->data[i]);
    }
//    matrix_mul(void* input, void* output,const float beta, h, w);
//    loglog(base_layer->data,  base_layer_log->data, c, h, w);

#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("3. Log map on base layer time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif

    // 4. Compress and fuse

#if test_time
    StartTime = 0;
                EndTime = 0;

                    TSCH = 0;
                    TSCL = 0;
                    StartTime = _itoll(TSCH, TSCL);
#endif


    float min_base_val = get_percentile(base_layer_log, min_percentile);
    float max_base_val = get_percentile(base_layer_log, max_percentile);



#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("4. Compress and fuse time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif

#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif
    ImageF *base_layer_compressed = create_image_f(w, h);
    for (int i = 0; i < size; ++i) {
        float val = base_layer_log->data[i];
        if (val < min_base_val) val = min_base_val;
        if (val > max_base_val) val = max_base_val;
        base_layer_compressed->data[i] = val;
    }

    ImageF *fused_image = create_image_f(w, h);
//    for (int i = 0; i < size; ++i) {
//        fused_image->data[i] = base_weight * base_layer_compressed->data[i] +
//                               detail_weight * detail_layer->data[i];
//    }
    matrix_add((void*) base_layer_compressed->data, (void*) detail_layer->data, (void*) fused_image->data, base_weight,detail_weight, h, w);
//    save_image_f_to_txt(fused_image, "../Images/fused_image.txt");
#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("4.5 time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif



#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif
    // 5. Normalize to 8-bit

    // 1. 找到最大值和最小值（这步不能省）
    float min_val = DBL_MAX;
    float max_val = -DBL_MAX;
    for (int i = 0; i < size; ++i) {
        float val = (float)fused_image->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

//    float min_val = fused_image->data[0];
//    float max_val = fused_image->data[0];
//
//    for (int i = 1; i < size; ++i) {
//        if (fused_image->data[i] < min_val) min_val = fused_image->data[i];
//        if (fused_image->data[i] > max_val) max_val = fused_image->data[i];
//    }

//    float min_val =  matrix_min(fused_image->data, h, w);
//    float max_val =  matrix_max(fused_image->data, h, w);

    // 2. 计算缩放系数（这步不能省）
    float range = max_val - min_val;
    if (range <= 1e-6) range = 1e-6; // 防止除以零
    float scale = 255.0f / range;

    // 3. 创建最终的8位图像
    ImageU8 *image_8bit = create_image_u8(w, h);

    float *temp_buffer = (float*)snet_malloc(size * sizeof(float), &g_startL2Address);
    g_startL2Address = USER_HEAP_ADDR;
    // 4. 一次性完成归一化和转换

    //    for (int i = 0; i < size; ++i) {
////        float normalized_val = (fused_image->data[i] - min_val) * scale;
//        image_8bit->data[i] = (unsigned char)(int)(temp_buffer[i] + 0.5);
//    }
    matrix_sub_mul(fused_image->data, temp_buffer, min_val, scale, h, w);// (fused_image->data[i] - min_val) * scale
    convert_float_char(temp_buffer, image_8bit->data, h, w);

#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("5. Normalize to 8-bit time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif
    save_image_u8_to_txt(image_8bit, "../Images/output.txt");
    return image_8bit;
}

// --- Main Execution Logic ---

void process_directory() {


    long long StartTime = 0;
    long long EndTime = 0;
#if test_time
    StartTime = 0;
    EndTime = 0;

    TSCH = 0;
    TSCL = 0;
    StartTime = _itoll(TSCH, TSCL);
#endif

    // --- Adjustable Parameters ---
    int guided_filter_radius = 16;
    float guided_filter_eps = 0.01f * 0.01f;
    float base_min_percentile = 0.005f;
    float base_max_percentile = 99.4f;
    float base_layer_weight = 0.5f;
    float detail_layer_weight = 1.0f;

    char raw_path[1024];
    ImageF *thermal_img_14bit = read_raw_thermal(raw_path);
    float max_pix = matrix_max((void*)thermal_img_14bit->data, thermal_img_14bit->height, thermal_img_14bit->width);

//    float max_pix = 0.0f;
//
//    for (int i = 0; i < thermal_img_14bit->width * thermal_img_14bit->height; ++i)
//    {
//        if (thermal_img_14bit->data[i] > max_pix)
//        {
//            max_pix = thermal_img_14bit->data[i];
//        }
//    }
#if test_time

    EndTime = _itoll(TSCH, TSCL);
    printf("0. time:%f ms\n", (EndTime - StartTime)/1000000.0);
#endif
//    ImageU8 *final_image;
    if (max_pix == 0)
    {
        ImageU8 *final_image = create_image_u8(thermal_img_14bit->width, thermal_img_14bit->height);
    }else
    {
        ImageU8 *final_image = enhanced_thermal_conversion(
                thermal_img_14bit,
                guided_filter_radius,
                guided_filter_eps,
                base_min_percentile,
                base_max_percentile,
                base_layer_weight,
                detail_layer_weight
        );
    }

}
int main()
{
    uint32_t ui_core_num = CSL_chipReadReg(CSL_CHIP_DNUM);

    if (ui_core_num == 0)
    {
        CACHE_setL1PSize(CACHE_L1_32KCACHE);
        CACHE_setL1DSize(CACHE_L1_32KCACHE);
        CACHE_setL2Size(CACHE_0KCACHE);
        CACHE_invAllL1p(CACHE_WAIT);
        //CGEM_regs->MAR[179]=0;
    }
    else
    {
        CACHE_setL1PSize(CACHE_L1_32KCACHE);
        CACHE_setL1DSize(CACHE_L1_32KCACHE);
        CACHE_setL2Size(CACHE_0KCACHE);
        CACHE_invAllL1p(CACHE_WAIT);
        //CGEM_regs->MAR[179]=0;
    }
    uint32_t DDR_start_addr = 0x82000000;
    uint32_t DDR_end_addr =   0xE2000000;
    set_cache(DDR_start_addr, DDR_end_addr);
    //core_work();
#if CALC_TOTAL_TIME
    long long StartTime = 0;
    long long EndTime = 0;
    TSCH = 0;
    TSCL = 0;
#endif

    int i_frame = 0;
    int step = 0;

    printf("--- Starting Enhanced Thermal Image Conversion (C Version) ---\n");
#if CALC_TOTAL_TIME
    if (i_frame == 0 && step == 0)
    {
        StartTime = _itoll(TSCH, TSCL);
    }
#endif
    process_directory();

#if CALC_TOTAL_TIME
    EndTime = _itoll(TSCH, TSCL);
    printf("total time:%f ms\n", (EndTime - StartTime) / 1000000.0);
#endif
    printf("\n--- All processing complete ---\n");

    return 0;
}




