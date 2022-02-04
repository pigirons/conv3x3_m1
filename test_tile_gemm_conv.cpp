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
    float *dst = (float*)alignAlloc(dst_h * dst_w * out_c * sizeof(float), 16);
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
            pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst);
    }

    gettimeofday(&start, NULL);
    for (i = 0; i < loop_time; i++)
    {
        conv_tile_gemm_f3s1_m1(src, src_h, src_w, in_c,
            pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst);
    }
    gettimeofday(&end, NULL);

    time = get_time(&start, &end) / loop_time;
    gflops = comp / time * 1e-9;

    printf("tile_gemm conv algo: time = %lfus, perf = %lf GFLOPS.\n", time * 1e6, gflops);

    memset(dst, 0, dst_h * dst_w * out_c * sizeof(float));
    conv_tile_gemm_f3s1_m1(src, src_h, src_w, in_c,
        pad_h, pad_w, cvt_filter, bias, out_c, buffer, dst);
    save_bin(dst, dst_h * dst_w * out_c, "tuned.bin");

    alignFree(src);
    alignFree(filter);
    alignFree(cvt_filter);
    alignFree(bias);
    alignFree(dst);
    alignFree(buffer);

    return 0;
}

