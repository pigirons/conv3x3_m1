#include <cstdio>
#include <cstdlib>

extern "C" {
void sgemm_kernel_m1_fp32_m8n12k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);

void sgemm_kernel_m1_fp32_m8n8k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);

void sgemm_kernel_m1_fp32_m8n4k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);

void sgemm_kernel_m1_fp32_m4n12k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);

void sgemm_kernel_m1_fp32_m4n8k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);

void sgemm_kernel_m1_fp32_m4n4k4(float *a_loc,
    float *b_loc,
    float *c_loc,
    int m,
    int n,
    int k);
}

void sgemm_kernel_fp32_m1(int m,
    int n,
    int k,
    float *a_loc,
    float *b_loc,
    float *c_loc)
{
    int m_left = m % 8;
    int n_left = n % 12;

    if (m_left == 0)
    {
        if (n_left == 0)
        {
            sgemm_kernel_m1_fp32_m8n12k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else if (n_left == 8)
        {
            sgemm_kernel_m1_fp32_m8n8k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else if (n_left == 4)
        {
            sgemm_kernel_m1_fp32_m8n4k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else
        {
            fprintf(stderr, "ERROR: n must be multiple of 4.\n");
            exit(0);
        }
    }
    else if (m_left == 4)
    {
        if (n_left == 0)
        {
            sgemm_kernel_m1_fp32_m4n12k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else if (n_left == 8)
        {
            sgemm_kernel_m1_fp32_m4n8k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else if (n_left == 4)
        {
            sgemm_kernel_m1_fp32_m4n4k4(a_loc, b_loc, c_loc, m, n, k);
        }
        else
        {
            fprintf(stderr, "ERROR: n must be multiple of 4.\n");
            exit(0);
        }
    }
    else
    {
        fprintf(stderr, "ERROR: m must be multiple of 4.\n");
        exit(0);
    }
}

