#include "conv_tile_gemm_f3s1_m1.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <sched.h>
#include <pthread.h>

static void* alignAlloc(size_t size, size_t alignment)
{
    void *p = NULL;
    if (0 == posix_memalign(&p, alignment, size))
    {
        return p;
    }
    return NULL;
}

static void alignFree(void *p)
{
    free(p);
}

static double get_time(struct timeval *start, struct timeval *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_usec - start->tv_usec) * 1e-6;
}

static void save_bin(float *a, int n, const char *file_name)
{
    FILE *fp = fopen(file_name, "wb");
    fwrite(a, sizeof(float), n, fp);
    fclose(fp);
}

static void conv_naive(float *src,
    int src_h,
    int src_w,
    int in_c,
    int pad_h,
    int pad_w,
    int out_c,
    float *filter,
    float *bias,
    float *dst)
{
    int src_pad_h = src_h + 2 * pad_h;
    int src_pad_w = src_w + 2 * pad_w;
    int dst_pad_h = src_pad_h - 3 + 1;
    int dst_pad_w = src_pad_w - 3 + 1;

    float *src_trans;
    if (pad_h == 0 && pad_w == 0)
    {
        src_trans = src;
    }
    else
    {
        src_trans = (float*)alignAlloc(src_pad_h * src_pad_w * in_c * sizeof(float), 16);
        float *p_src = src_trans;
        if (pad_h > 0)
        {
            for (int i = 0; i < in_c; i++)
            {
                memset(p_src, 0, pad_h * src_pad_w * sizeof(float));
                p_src += pad_h * src_pad_w;

                for (int j = 0; j < src_h; j++)
                {
                    for (int k = 0; k < pad_w; k++)
                    {
                        *p_src++ = 0.0f;
                    }

                    memcpy(p_src, src, src_w * sizeof(float));
                    p_src += src_w;
                    src += src_w;

                    for (int k = 0; k < pad_w; k++)
                    {
                        *p_src++ = 0.0f;
                    }
                }

                memset(p_src, 0, pad_h * src_pad_w * sizeof(float));
                p_src += pad_h * src_pad_w;
            }
        }
    }

    for (int i = 0; i < out_c; i++)
    {
        for (int j = 0; j < dst_pad_h; j++)
        {
            for (int k = 0; k < dst_pad_w; k++)
            {
                float *p_src = src_trans + j * src_pad_w + k;
                float rst = bias[i];

                for (int ii = 0; ii < in_c; ii++)
                {
                    rst += p_src[ii * src_pad_h * src_pad_w + 0 * src_pad_w + 0] * filter[ii * 9 + 0];
                    rst += p_src[ii * src_pad_h * src_pad_w + 0 * src_pad_w + 1] * filter[ii * 9 + 1];
                    rst += p_src[ii * src_pad_h * src_pad_w + 0 * src_pad_w + 2] * filter[ii * 9 + 2];
                    rst += p_src[ii * src_pad_h * src_pad_w + 1 * src_pad_w + 0] * filter[ii * 9 + 3];
                    rst += p_src[ii * src_pad_h * src_pad_w + 1 * src_pad_w + 1] * filter[ii * 9 + 4];
                    rst += p_src[ii * src_pad_h * src_pad_w + 1 * src_pad_w + 2] * filter[ii * 9 + 5];
                    rst += p_src[ii * src_pad_h * src_pad_w + 2 * src_pad_w + 0] * filter[ii * 9 + 6];
                    rst += p_src[ii * src_pad_h * src_pad_w + 2 * src_pad_w + 1] * filter[ii * 9 + 7];
                    rst += p_src[ii * src_pad_h * src_pad_w + 2 * src_pad_w + 2] * filter[ii * 9 + 8];
                }
                dst[i * dst_pad_h * dst_pad_w + j * dst_pad_w + k] = rst;
            }
        }
        filter += 3 * 3 * in_c;
    }
    
    if (src_trans != src)
    {
        alignFree(src_trans);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        fprintf(stderr, "Usage: %s src_h src_w in_c out_c pad_h pad_w bind_cpu\n", argv[0]);
        exit(0);
    }
    int src_h = atoi(argv[1]);
    int src_w = atoi(argv[2]);
    int in_c  = atoi(argv[3]);
    int out_c = atoi(argv[4]);
    int pad_h = atoi(argv[5]);
    int pad_w = atoi(argv[6]);
    if (strncmp(argv[7], "p", 1) == 0)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    else if (strncmp(argv[7], "e", 1) == 0)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
    }

    int dst_h = src_h + pad_h * 2 - 3 + 1;
    int dst_w = src_w + pad_w * 2 - 3 + 1;

    int buf_size = conv_tile_gemm_f3s1_buf_size_m1(src_h, src_w, in_c, pad_h, pad_w, out_c);
    int cvt_filter_size = conv_tile_gemm_f3s1_cvt_filter_size_m1(in_c, out_c);

    struct timeval start, end;
    long long comp = 2LL * 3 * 3 * dst_h * dst_w * in_c * out_c;
    int loop_time = (int)(1e11 / comp) + 1;
    printf("fp32 computing count = %lld, loop_time = %d.\n", comp, loop_time);
    double time, gflops;
    int i;

    float *src = (float*)alignAlloc(src_h * src_w * in_c * sizeof(float), 16);
    float *filter = (float*)alignAlloc(3 * 3 * in_c * out_c * sizeof(float), 16);
    float *cvt_filter = (float*)alignAlloc(cvt_filter_size, 16);
    float *bias = (float*)alignAlloc(out_c * sizeof(float), 16);
    float *dst1 = (float*)alignAlloc(dst_h * dst_w * out_c * sizeof(float), 16);
    float *dst2 = (float*)alignAlloc(dst_h * dst_w * out_c * sizeof(float), 16);
    void *buffer = alignAlloc(buf_size, 16);

    for (i = 0; i < src_h * src_w * in_c; i++)
    {
        src[i] = (float)rand() / RAND_MAX;
    }
    for (i = 0; i < 3 * 3 * in_c * out_c; i++)
    {
        filter[i] = (float)rand() / RAND_MAX;
    }
    for (i = 0; i < out_c; i++)
    {
        bias[i] = (float)rand() / RAND_MAX;
    }
    conv_tile_gemm_f3s1_cvt_filter_m1(filter, in_c, out_c, cvt_filter);

    // warm up
    for (i = 0; i < loop_time; i++)
    {
        conv_tile_gemm_f3s1_m1(src, src_h, src_w, in_c,
            pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst2);
    }

    gettimeofday(&start, NULL);
    for (i = 0; i < loop_time; i++)
    {
        conv_tile_gemm_f3s1_m1(src, src_h, src_w, in_c,
            pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst2);
    }
    gettimeofday(&end, NULL);

    time = get_time(&start, &end) / loop_time;
    gflops = comp / time * 1e-9;

    printf("tile_gemm conv algo: time = %lfus, perf = %lf GFLOPS.\n", time * 1e6, gflops);

    conv_naive(src, src_h, src_w, in_c, pad_h, pad_w, out_c, filter, bias, dst1);
    save_bin(dst1, dst_h * dst_w * out_c, "naive.bin");
    
    memset(dst2, 0, dst_h * dst_w * out_c * sizeof(float));
    conv_tile_gemm_f3s1_m1(src, src_h, src_w, in_c,
        pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst2);
    save_bin(dst2, dst_h * dst_w * out_c, "tuned.bin");

    alignFree(src);
    alignFree(filter);
    alignFree(cvt_filter);
    alignFree(bias);
    alignFree(dst1);
    alignFree(dst2);
    alignFree(buffer);

    return 0;
}

