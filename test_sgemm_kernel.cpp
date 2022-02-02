#define _GNU_SOURCE

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <sched.h>
#include <pthread.h>

void sgemm_kernel_fp32_m1(float *,
    float *,
    float *,
    int,
    int,
    int);

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

static void trans_a(float *src, int m, int k, float *dst)
{
    int i, j;
    for (i = 0; i <= m - 8; i += 8)
    {
        float *ps = src + i * k;
        float *pd = dst + i * k;
        for (j = 0; j < k; j++)
        {
            pd[j * 8 + 0] = ps[0 * k + j];
            pd[j * 8 + 1] = ps[1 * k + j];
            pd[j * 8 + 2] = ps[2 * k + j];
            pd[j * 8 + 3] = ps[3 * k + j];
            pd[j * 8 + 4] = ps[4 * k + j];
            pd[j * 8 + 5] = ps[5 * k + j];
            pd[j * 8 + 6] = ps[6 * k + j];
            pd[j * 8 + 7] = ps[7 * k + j];
        }
    }
    if (m - i == 4)
    {
        float *ps = src + i * k;
        float *pd = dst + i * k;
        for (j = 0; j < k; j++)
        {
            pd[j * 4 + 0] = ps[0 * k + j];
            pd[j * 4 + 1] = ps[1 * k + j];
            pd[j * 4 + 2] = ps[2 * k + j];
            pd[j * 4 + 3] = ps[3 * k + j];
        }
    }
    else if (m != i)
    {
        fprintf(stderr, "ERROR: m must be multiple of 4.\n");
        exit(0);
    }
}

static void trans_b(float *src, int k, int n, float *dst)
{
    int i, j;
    for (i = 0; i <= n - 12; i += 12)
    {
        float *ps = src + i;
        for (j = 0; j < k; j++)
        {
            memcpy(dst, ps, 12 * sizeof(float));
            ps += n;
            dst += 12;
        }
    }
    if (n - i == 8)
    {
        float *ps = src + i;
        for (j = 0; j < k; j++)
        {
            memcpy(dst, ps, 8 * sizeof(float));
            ps += n;
            dst += 8;
        }
    }
    else if (n - i == 4)
    {
        float *ps = src + i;
        for (j = 0; j < k; j++)
        {
            memcpy(dst, ps, 4 * sizeof(float));
            ps += n;
            dst += 4;
        }
    }
    else if (n != i)
    {
        fprintf(stderr, "ERROR: n must be multiple of 4.\n");
        exit(0);
    }
}

static void trans_c(float *src, int m, int n, float *dst)
{
    int i, j, ii;
    for (i = 0; i <= m - 8; i += 8)
    {
        for (j = 0; j <= n - 12; j += 12)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 8; ii++)
            {
                memcpy(pd, src, 12 * sizeof(float));
                src += 12;
                pd += n;
            }
        }
        if (n - j == 8)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 8; ii++)
            {
                memcpy(pd, src, 8 * sizeof(float));
                src += 8;
                pd += n;
            }
        }
        else if (n - j == 4)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 8; ii++)
            {
                memcpy(pd, src, 4 * sizeof(float));
                src += 4;
                pd += n;
            }
        }
        else if (n != j)
        {
            fprintf(stderr, "ERROR: n must be multiple of 4.\n");
            exit(0);
        }
    }
    if (m - i == 4)
    {
        for (j = 0; j <= n - 12; j += 12)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 4; ii++)
            {
                memcpy(pd, src, 12 * sizeof(float));
                src += 12;
                pd += n;
            }
        }
        if (n - j == 8)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 4; ii++)
            {
                memcpy(pd, src, 8 * sizeof(float));
                src += 8;
                pd += n;
            }
        }
        else if (n - j == 4)
        {
            float *pd = dst + i * n + j;
            for (ii = 0; ii < 4; ii++)
            {
                memcpy(pd, src, 4 * sizeof(float));
                src += 4;
                pd += n;
            }
        }
        else if (n != j)
        {
            fprintf(stderr, "ERROR: n must be multiple of 4.\n");
            exit(0);
        }
    }
    else if (m != i)
    {
        fprintf(stderr, "ERROR: m must be multiple of 4.\n");
        exit(0);
    }
}

static void sgemm_naive(float *a, float *b, float *c, int m, int n, int k)
{
    int i, j, kk;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            float rst = 0.0f;
            for (kk = 0; kk < k; kk++)
            {
                rst += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = rst;
        }
    }
}

static void save_bin(float *a, int n, const char *file_name)
{
    FILE *fp = fopen(file_name, "wb");
    fwrite(a, sizeof(float), n, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s m n k bind_cpu\n", argv[0]);
        exit(0);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    if (strncmp(argv[4], "p", 1) == 0)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    else if (strncmp(argv[4], "e", 1) == 0)
    {
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
    }

    struct timeval start, end;
    long long comp = 2 * m * n * k;
    int loop_time = (int)(1e11 / comp) + 1;
    double time, gflops;
    int i;

    float *a, *b, *c;
    float *at, *bt, *ct;

    a = (float*)alignAlloc(m * k * sizeof(float), 16);
    b = (float*)alignAlloc(k * n * sizeof(float), 16);
    c = (float*)alignAlloc(m * n * sizeof(float), 16);

    for (i = 0; i < m * k; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
    }
    for (i = 0; i < k * n; i++)
    {
        b[i] = (float)rand() / RAND_MAX;
    }

    at = (float*)alignAlloc(m * k * sizeof(float), 16);
    bt = (float*)alignAlloc(k * n * sizeof(float), 16);
    ct = (float*)alignAlloc(m * n * sizeof(float), 16);

    trans_a(a, m, k, at);
    trans_b(b, k, n, bt);

    // warm up
    for (i = 0; i < loop_time; i++)
    {
        sgemm_kernel_fp32_m1(at, bt, ct, m, n, k);
    }

    gettimeofday(&start, NULL);
    for (i = 0; i < loop_time; i++)
    {
        sgemm_kernel_fp32_m1(at, bt, ct, m, n, k);
    }
    gettimeofday(&end, NULL);

    time = get_time(&start, &end) / loop_time;
    gflops = comp / time * 1e-9;

    printf("sgemm_kernel m = %d, n = %d, k = %d, time = %lfus, perf = %lf GFLOPS.\n", m, n, k, time * 1e6, gflops);

    sgemm_naive(a, b, c, m, n, k);
    save_bin(c, m * n, "naive.bin");

    memset(ct, 0, m * n * sizeof(float));
    sgemm_kernel_fp32_m1(at, bt, ct, m, n, k);
    trans_c(ct, m, n, c);
    save_bin(c, m * n, "tuned.bin");

    alignFree(a);
    alignFree(b);
    alignFree(c);
    alignFree(at);
    alignFree(bt);
    alignFree(ct);

    return 0;
}

