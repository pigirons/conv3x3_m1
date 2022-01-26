#ifndef _CONV_TILE_GEMM_H
#define _CONV_TILE_GEMM_H

int conv_tile_gemm_f3s1_buf_size_m1(int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    int num_outs);

int conv_tile_gemm_f3s1_cvt_filter_size_m1(int channels,
    int num_outs);

void conv_tile_gemm_f3s1_cvt_filter_m1(float *filters,
    int channels,
    int num_outs,
    float *filters_cvt);

void conv_tile_gemm_f3s1_m1(float *src,
    int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    float *filters_cvt,
    float *bias,
    int num_outs,
    void *buffer,
    float *dst);

#endif

