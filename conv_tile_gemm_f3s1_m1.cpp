#include "conv_tile_gemm_f3s1_m1.h"
#include "conv_tile_gemm_f3s1_params.h"

#include <arm_neon.h>
#include <cstring>
#include <cstdio>

#define MAX_INT(A, B) ((A) < (B) ? (B) : (A))
#define MIN_INT(A, B) ((A) < (B) ? (A) : (B))
#define TG_PADDING(SIZE, ALIGN) (((SIZE) + (ALIGN) - 1) / (ALIGN) * (ALIGN))
#define TRANSPOSE_4X4(Q0, Q1, Q2, Q3) \
{ \
    float32x4x2_t __Q0, __Q1; \
    __Q0 = vtrnq_f32(Q0, Q1); \
    __Q1 = vtrnq_f32(Q2, Q3); \
    Q0 = vcombine_f32(vget_low_f32(__Q0.val[0]), \
        vget_low_f32(__Q1.val[0])); \
    Q1 = vcombine_f32(vget_low_f32(__Q0.val[1]), \
        vget_low_f32(__Q1.val[1])); \
    Q2 = vcombine_f32(vget_high_f32(__Q0.val[0]), \
        vget_high_f32(__Q1.val[0])); \
    Q3 = vcombine_f32(vget_high_f32(__Q0.val[1]), \
        vget_high_f32(__Q1.val[1])); \
}

#define CONV_TG_SRC_TRANS_ZERO_W12(DST, LINES) \
{ \
    int idx; \
    for (idx = 0; idx < (LINES); idx++) \
    { \
        vst1q_f32((DST) + idx * CONV_TG_REG_N_SIZE + 0, vzero); \
        vst1q_f32((DST) + idx * CONV_TG_REG_N_SIZE + 4, vzero); \
        vst1q_f32((DST) + idx * CONV_TG_REG_N_SIZE + 8, vzero); \
    } \
}

#define CONV_TG_SRC_TRANS_ZERO_W8(DST, LINES) \
{ \
    int idx; \
    for (idx = 0; idx < (LINES); idx++) \
    { \
        vst1q_f32((DST) + idx * 8 + 0, vzero); \
        vst1q_f32((DST) + idx * 8 + 4, vzero); \
    } \
}

#define CONV_TG_SRC_TRANS_ZERO_W4(DST, LINES) \
{ \
    int idx; \
    for (idx = 0; idx < (LINES); idx++) \
    { \
        vst1q_f32((DST) + idx * 4 + 0, vzero); \
    } \
}

extern void sgemm_kernel_fp32_m1(int m,
    int n,
    int k,
    float *a_loc,
    float *b_loc,
    float *c_loc);

int conv_tile_gemm_f3s1_buf_size_m1(int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    int num_outs)
{
    int src_pad_h = CONV_TG_REG_N_SIZE - 1 + 3;
    int src_pad_w = CONV_TG_REG_N_SIZE - 1 + 3;

    int flt_size = 3 * 3;
    int chn_size = CONV_TG_F3_CHN_SIZE;

    int k = flt_size * chn_size;
    int kt_size = TG_PADDING(k, 4);
    int c_div = channels / chn_size;
    int c_rem = channels - c_div * chn_size;

    int k_pad = c_div * kt_size +
        TG_PADDING(c_rem * flt_size, 4);
    int m_pad = MIN_INT(18 * CONV_TG_REG_M_SIZE,
        TG_PADDING(num_outs, 4));
    int n_pad = CONV_TG_REG_N_SIZE * CONV_TG_REG_N_SIZE;

    int buffer_size = MAX_INT(CONV_TG_REG_M_SIZE, flt_size) *
        n_pad * sizeof(float);
    int src_pad_size = channels * src_pad_h * src_pad_w * sizeof(float);
    int src_trans_size = k_pad * n_pad * sizeof(float);
    int dst_trans_size = m_pad * n_pad * sizeof(float);

    return buffer_size + src_trans_size +
        MAX_INT(src_pad_size, dst_trans_size);
}

static void conv_tg_flt_one_blk(float *cvt_filters,
    int stride,
    int k,
    int num_outs,
    float *filters_cvt)
{
    int i, j, ii, jj;

    float32x4_t vr[8];
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (i = 0; i <= num_outs - 8; i += 8)
    {
        for (j = 0; j <= k - 4; j += 4)
        {
            float *cfd = cvt_filters +
                i * stride + j;

            vr[0] = vld1q_f32(cfd + 0 * stride);
            vr[1] = vld1q_f32(cfd + 1 * stride);
            vr[2] = vld1q_f32(cfd + 2 * stride);
            vr[3] = vld1q_f32(cfd + 3 * stride);
            vr[4] = vld1q_f32(cfd + 4 * stride);
            vr[5] = vld1q_f32(cfd + 5 * stride);
            vr[6] = vld1q_f32(cfd + 6 * stride);
            vr[7] = vld1q_f32(cfd + 7 * stride);
            TRANSPOSE_4X4(vr[0], vr[1], vr[2], vr[3]);
            TRANSPOSE_4X4(vr[4], vr[5], vr[6], vr[7]);
            vst1q_f32(filters_cvt + 0, vr[0]);
            vst1q_f32(filters_cvt + 4, vr[4]);
            vst1q_f32(filters_cvt + 8, vr[1]);
            vst1q_f32(filters_cvt + 12, vr[5]);
            vst1q_f32(filters_cvt + 16, vr[2]);
            vst1q_f32(filters_cvt + 20, vr[6]);
            vst1q_f32(filters_cvt + 24, vr[3]);
            vst1q_f32(filters_cvt + 28, vr[7]);

            filters_cvt += 32;
        }
        if (j < k)
        {
            float *cfd = cvt_filters +
                i * stride + j;

            for (ii = 0; ii < k - j; ii++)
            {
                for (jj = 0; jj < 8; jj++)
                {
                    filters_cvt[jj] = cfd[jj * stride + ii];
                }
                filters_cvt += 8;
            }
            for (; ii < 4; ii++)
            {
                vst1q_f32(filters_cvt + 0, vzero);
                vst1q_f32(filters_cvt + 4, vzero);
                filters_cvt += 8;
            }
        }
    }
    if (num_outs - i > 4)
    {
        int k_pad = TG_PADDING(k, 4);

        for (j = 0; j < k; j++)
        {
            float *cfd = cvt_filters +
                i * stride + j;

            for (ii = 0; ii < num_outs - i; ii++)
            {
                filters_cvt[ii] = cfd[ii * stride];
            }
            for (; ii < 8; ii++)
            {
                filters_cvt[ii] = 0.0f;
            }
            filters_cvt += 8;
        }
        for (; j < k_pad; j++)
        {
            vst1q_f32(filters_cvt + 0, vzero);
            vst1q_f32(filters_cvt + 4, vzero);
            filters_cvt += 8;
        }
    }
    else if (i < num_outs)
    {
        int k_pad = TG_PADDING(k, 4);

        for (j = 0; j < k; j++)
        {
            float *cfd = cvt_filters +
                i * stride + j;

            for (ii = 0; ii < num_outs - i; ii++)
            {
                filters_cvt[ii] = cfd[ii * stride];
            }
            for (; ii < 4; ii++)
            {
                filters_cvt[ii] = 0.0f;
            }
            filters_cvt += 4;
        }
        for (; j < k_pad; j++)
        {
            vst1q_f32(filters_cvt + 0, vzero);
            filters_cvt += 4;
        }
    }
}

static void conv_tg_blocking_cvt_filter(float *filters,
    int channels,
    int num_outs,
    int m_blk,
    int chn_blk,
    float *filters_cvt)
{
    int i, j;

    int flt_size = 3 * 3;
    int mt_size = m_blk;
    int chn_size = chn_blk;
    int k = channels * flt_size;

    for (i = 0; i <= num_outs - mt_size; i += mt_size)
    {
        for (j = 0; j <= channels - chn_size; j += chn_size)
        {
            int kt_size = chn_size * flt_size;
            float *flt_d = filters + i * k + j * flt_size;
            conv_tg_flt_one_blk(flt_d, k, kt_size, mt_size,
                filters_cvt);

            filters_cvt += mt_size * TG_PADDING(kt_size, 4);
        }
        if (j < channels)
        {
            int kt_size = (channels - j) * flt_size;
            float *flt_d = filters + i * k + j * flt_size;
            conv_tg_flt_one_blk(flt_d, k, kt_size, mt_size,
                filters_cvt);

            filters_cvt += mt_size * TG_PADDING(kt_size, 4);
        }
    }
    if (i < num_outs)
    {
        mt_size = num_outs - i;
        for (j = 0; j <= channels - chn_size; j += chn_size)
        {
            int kt_size = chn_size * flt_size;
            float *flt_d = filters + i * k + j * flt_size;
            conv_tg_flt_one_blk(flt_d, k, kt_size,
                mt_size, filters_cvt);

            filters_cvt += TG_PADDING(mt_size, 4) *
                TG_PADDING(kt_size, 4);
        }
        if (j < channels)
        {
            int kt_size = (channels - j) * flt_size;
            float *flt_d = filters + i * k + j * flt_size;
            conv_tg_flt_one_blk(flt_d, k, kt_size, mt_size,
                filters_cvt);
        }
    }
}

int conv_tile_gemm_f3s1_cvt_filter_size_m1(int channels,
    int num_outs)
{
    int flt_size = 3 * 3;
    int chn_size = CONV_TG_F3_CHN_SIZE;

    int k = flt_size * chn_size;
    int kt_size = TG_PADDING(k, 4);
    int c_div = channels / chn_size;
    int c_rem = channels - c_div * chn_size;

    int k_pad = c_div * kt_size +
        TG_PADDING(c_rem * flt_size, 4);
    int m_pad = TG_PADDING(num_outs, 4);

    return k_pad * m_pad * sizeof(float);
}

void conv_tile_gemm_f3s1_cvt_filter_m1(float *filters,
    int channels,
    int num_outs,
    float *filters_cvt)
{
    int flt_size = 3 * 3;
    int chn_size = CONV_TG_F3_CHN_SIZE;
    int k = flt_size * chn_size;
    int mt_size = 18 * CONV_TG_REG_M_SIZE;
    int kt_size = TG_PADDING(k, 4);

    conv_tg_blocking_cvt_filter(filters, channels, num_outs, mt_size,
        chn_size, filters_cvt);
}

static inline bool conv_tg_blocking_image(float *src,
    int src_h,
    int src_w,
    int channels,
    int hori_beg,
    int hori_end,
    int vert_beg,
    int vert_end,
    int &h_idx,
    int &w_idx,
    float *dst,
    int &real_dst_h,
    int &real_dst_w)
{
    int i, j, k;

    if (h_idx + 3 - 1 >= vert_end)
    {
        return false;
    }

    real_dst_h = MIN_INT(vert_end - h_idx,
        CONV_TG_F3S1_SRC_SIZE);
    real_dst_w = MIN_INT(hori_end - w_idx,
        CONV_TG_F3S1_SRC_SIZE);

    int h_beg = MAX_INT(h_idx, 0);
    int h_end = MIN_INT(h_idx + real_dst_h, src_h);

    int w_beg = MAX_INT(w_idx, 0);
    int w_end = MIN_INT(w_idx + real_dst_w, src_w);

    int real_src_h = h_end - h_beg;
    int real_src_w = w_end - w_beg;

    int h_dst_beg = MAX_INT(-h_idx, 0);
    int w_dst_beg = MAX_INT(-w_idx, 0);

    for (i = 0; i < channels; i++)
    {
        float *src_d = src + h_beg * src_w + w_beg;

        memset(dst, 0, h_dst_beg * real_dst_w * sizeof(float));
        dst += h_dst_beg * real_dst_w;

        for (j = h_beg; j < h_end; j++)
        {
            for (k = 0; k < w_dst_beg; k++)
            {
                dst[k] = 0.0f;
            }

            memcpy(dst + w_dst_beg, src_d, real_src_w * sizeof(float));

            for (k = w_dst_beg + real_src_w; k < real_dst_w; k++)
            {
                dst[k] = 0.0f;
            }

            dst += real_dst_w;
            src_d += src_w;
        }

        memset(dst, 0, (real_dst_h - h_dst_beg - real_src_h) *
            real_dst_w * sizeof(float));
        dst += (real_dst_h - h_dst_beg - real_src_h) * real_dst_w;

        src += src_w * src_h;
    }

    if (w_idx + real_dst_w >= hori_end)
    {
        w_idx = hori_beg;
        h_idx += (real_dst_h - 3 + 1);
    }
    else
    {
        w_idx += (real_dst_w - 3 + 1);
    }
    return true;
}

#define CONV_TG_F3S1_SRC_TRANS_W12(SRC, DST) \
{ \
    vr[0] = vld1q_f32((SRC) + 0); \
    vr[1] = vld1q_f32((SRC) + 4); \
    vr[2] = vld1q_f32((SRC) + 8); \
    vst1q_f32((DST) + 0, vr[0]); \
    vst1q_f32((DST) + 4, vr[1]); \
    vst1q_f32((DST) + 8, vr[2]); \
}

#define CONV_TG_F3S1_SRC_TRANS_W8(SRC, DST) \
{ \
    vr[0] = vld1q_f32((SRC) + 0); \
    vr[1] = vld1q_f32((SRC) + 4); \
    vst1q_f32((DST) + 0, vr[0]); \
    vst1q_f32((DST) + 4, vr[1]); \
}

#define CONV_TG_F3S1_SRC_TRANS_W4(SRC, DST) \
{ \
    vr[0] = vld1q_f32((SRC) + 0); \
    vst1q_f32((DST) + 0, vr[0]); \
}

static void conv_tg_f3s1_src_trans_wmax(float *src_pad,
    int src_pad_h,
    int channels,
    float *src_trans)
{
    int i, j, kk;

    int k = 3 * 3 * channels;
    int k_pad = TG_PADDING(k, 4);
    int tail = k_pad - k;

    int src_trans_h = src_pad_h - 3 + 1;
    int src_pad_align = src_pad_h * CONV_TG_F3S1_SRC_SIZE;

    float32x4_t vr[3];
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (i = 0; i < src_trans_h; i++)
    {
        float *src_pad_d = src_pad;

        for (j = 0; j < channels; j++)
        {
            for (kk = 0; kk < 3; kk++)
            {
                CONV_TG_F3S1_SRC_TRANS_W12(
                    src_pad_d + 0 + CONV_TG_F3S1_SRC_SIZE * kk,
                    src_trans + CONV_TG_REG_N_SIZE * 0);
                CONV_TG_F3S1_SRC_TRANS_W12(
                    src_pad_d + 1 + CONV_TG_F3S1_SRC_SIZE * kk,
                    src_trans + CONV_TG_REG_N_SIZE * 1);
                CONV_TG_F3S1_SRC_TRANS_W12(
                    src_pad_d + 2 + CONV_TG_F3S1_SRC_SIZE * kk,
                    src_trans + CONV_TG_REG_N_SIZE * 2);

                src_trans += 3 * CONV_TG_F3S1_DST_SIZE;
            }

            src_pad_d += src_pad_align;
        }
        CONV_TG_SRC_TRANS_ZERO_W12(src_trans, tail);

        src_trans += tail * CONV_TG_REG_N_SIZE;
        src_pad += CONV_TG_F3S1_SRC_SIZE;
    }
}

static inline void conv_tg_line_copy(float *src,
    int len,
    float *dst)
{
    int i;
    for (i = 0; i <= len - 4; i += 4)
    {
        vst1q_f32(dst + i, vld1q_f32(src + i));
    }
    for (; i < len; i++)
    {
        dst[i] = src[i];
    }
}

static void conv_tg_f3s1_src_trans(float *src_pad,
    int src_pad_h,
    int src_pad_w,
    int channels,
    float *buffer,
    float *src_trans)
{
    int i, j, kk;

    int k = 3 * 3 * channels;
    int k_pad = TG_PADDING(k, 4);
    int tail = k_pad - k;

    int src_trans_h = src_pad_h - 3 + 1;
    int src_trans_w = src_pad_w - 3 + 1;

    int n = src_trans_h * src_trans_w;
    int n_pad = TG_PADDING(n, 4);

    int src_pad_align = src_pad_h * src_pad_w;
    int src_trans_align = channels * 9 * CONV_TG_REG_N_SIZE;

    float32x4_t vr[3];
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (i = 0; i < 9; i++)
    {
        for (j = n; j < n_pad; j++)
        {
            buffer[i * n_pad + j] = 0.0f;
        }
    }

    for (i = 0; i < channels; i++)
    {
        float *src_pad_d = src_pad;
        float *buf_d = buffer;

        for (j = 0; j < src_trans_h; j++)
        {
            for (kk = 0; kk < 3; kk++)
            {
                conv_tg_line_copy(src_pad_d + kk * src_pad_w + 0,
                    src_trans_w, buf_d + (kk * 3 + 0) * n_pad);
                conv_tg_line_copy(src_pad_d + kk * src_pad_w + 1,
                    src_trans_w, buf_d + (kk * 3 + 1) * n_pad);
                conv_tg_line_copy(src_pad_d + kk * src_pad_w + 2,
                    src_trans_w, buf_d + (kk * 3 + 2) * n_pad);
            }

            src_pad_d += src_pad_w;
            buf_d += src_trans_w;
        }

        for (j = 0; j <= n_pad - CONV_TG_REG_N_SIZE;
            j += CONV_TG_REG_N_SIZE)
        {
            float *st_d = src_trans + j * k_pad +
                i * 9 * CONV_TG_REG_N_SIZE;

            for (kk = 0; kk < 9; kk++)
            {
                CONV_TG_F3S1_SRC_TRANS_W12(buffer + n_pad * kk + j,
                    st_d + kk * CONV_TG_REG_N_SIZE);
            }
        }
        if (n_pad - j == 8)
        {
            float *st_d = src_trans + j * k_pad +
                i * 9 * 8;

            for (kk = 0; kk < 9; kk++)
            {
                CONV_TG_F3S1_SRC_TRANS_W8(buffer + n_pad * kk + j,
                    st_d + kk * 8);
            }
        }
        else if (n_pad - j == 4)
        {
            float *st_d = src_trans + j * k_pad +
                i * 9 * 4;

            for (kk = 0; kk < 9; kk++)
            {
                CONV_TG_F3S1_SRC_TRANS_W4(buffer + n_pad * kk + j,
                    st_d + kk * 4);
            }
        }

        src_pad += src_pad_align;
    }
    if (tail > 0)
    {
        for (i = 0; i <= n_pad - CONV_TG_REG_N_SIZE;
            i += CONV_TG_REG_N_SIZE)
        {
            float *dst_d = src_trans + i * k_pad +
                k * CONV_TG_REG_N_SIZE;

            CONV_TG_SRC_TRANS_ZERO_W12(dst_d, tail);
        }
        if (n_pad - i == 8)
        {
            float *dst_d = src_trans + i * k_pad + k * 8;

            CONV_TG_SRC_TRANS_ZERO_W8(dst_d, tail);
        }
        else if (n_pad - i == 4)
        {
            float *dst_d = src_trans + i * k_pad + k * 4;

            CONV_TG_SRC_TRANS_ZERO_W4(dst_d, tail);
        }
    }
}

#define CONV_TG_F3S1_DST_TRANS_W12(SRC, LDS, IDX, DST, LDD) \
{ \
    vr[0] = vld1q_f32((SRC) + (LDS) * (IDX) + 0); \
    vr[1] = vld1q_f32((SRC) + (LDS) * (IDX) + 4); \
    vr[2] = vld1q_f32((SRC) + (LDS) * (IDX) + 8); \
    vr[0] = vaddq_f32(vr[0], vb[(IDX)]); \
    vr[1] = vaddq_f32(vr[1], vb[(IDX)]); \
    vr[2] = vaddq_f32(vr[2], vb[(IDX)]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 0, vr[0]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 4, vr[1]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 8, vr[2]); \
}

#define CONV_TG_F3S1_DST_TRANS_W8(SRC, LDS, IDX, DST, LDD) \
{ \
    vr[0] = vld1q_f32((SRC) + (LDS) * (IDX) + 0); \
    vr[1] = vld1q_f32((SRC) + (LDS) * (IDX) + 4); \
    vr[0] = vaddq_f32(vr[0], vb[(IDX)]); \
    vr[1] = vaddq_f32(vr[1], vb[(IDX)]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 0, vr[0]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 4, vr[1]); \
}

#define CONV_TG_F3S1_DST_TRANS_W4(SRC, LDS, IDX, DST, LDD) \
{ \
    vr[0] = vld1q_f32((SRC) + (LDS) * (IDX) + 0); \
    vr[0] = vaddq_f32(vr[0], vb[(IDX)]); \
    vst1q_f32((DST) + (IDX) * (LDD) + 0, vr[0]); \
}

static void conv_tg_f3s1_dst_trans_wmax(float *dst_trans,
    int dst_trans_h,
    int num_outs,
    float *bias,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j, k;

    float32x4_t vr[3];
    float32x4_t vb[CONV_TG_REG_M_SIZE];

    int dst_align = dst_h * dst_w;

    for (i = 0; i <= num_outs - CONV_TG_REG_M_SIZE;
        i += CONV_TG_REG_M_SIZE)
    {
        float *dst_d = dst;
        for (j = 0; j < CONV_TG_REG_M_SIZE; j++)
        {
            vb[j] = vdupq_n_f32(bias[i + j]);
        }
        for (j = 0; j < dst_trans_h; j++)
        {
            for (k = 0; k < CONV_TG_REG_M_SIZE; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W12(dst_trans,
                    CONV_TG_REG_N_SIZE, k, dst_d, dst_align);
            }

            dst_trans += CONV_TG_REG_M_SIZE * CONV_TG_REG_N_SIZE;
            dst_d += dst_w;
        }
        dst += dst_align * CONV_TG_REG_M_SIZE;
    }
    if (i < num_outs)
    {
        float *dst_d = dst;
        for (j = 0; j < num_outs - i; j++)
        {
            vb[j] = vdupq_n_f32(bias[i + j]);
        }
        for (j = 0; j < dst_trans_h; j++)
        {
            for (k = 0; k < num_outs - i; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W12(dst_trans,
                    CONV_TG_REG_N_SIZE, k, dst_d, dst_align);
            }

            dst_trans += TG_PADDING(num_outs - i, 4) *
                CONV_TG_REG_N_SIZE;
            dst_d += dst_w;
        }
    }
}

static void conv_tg_f3s1_dst_trans(float *dst_trans,
    int dst_trans_h,
    int dst_trans_w,
    int num_outs,
    float *bias,
    float *buffer,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j, k;

    float32x4_t vr[3];
    float32x4_t vb[CONV_TG_REG_M_SIZE];

    int n = dst_trans_h * dst_trans_w;
    int n_pad = TG_PADDING(n, 4);

    int dst_align = dst_h * dst_w;

    for (i = 0; i <= num_outs - CONV_TG_REG_M_SIZE;
        i += CONV_TG_REG_M_SIZE)
    {
        for (j = 0; j < CONV_TG_REG_M_SIZE; j++)
        {
            vb[j] = vdupq_n_f32(bias[i + j]);
        }

        float *buf_d = buffer;
        for (j = 0; j <= n_pad - CONV_TG_REG_N_SIZE;
            j += CONV_TG_REG_N_SIZE)
        {
            for (k = 0; k < CONV_TG_REG_M_SIZE; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W12(dst_trans,
                    CONV_TG_REG_N_SIZE, k, buf_d, n_pad);
            }

            buf_d += CONV_TG_REG_N_SIZE;
            dst_trans += CONV_TG_REG_M_SIZE *
                CONV_TG_REG_N_SIZE;
        }
        if (n_pad - j == 8)
        {
            for (k = 0; k < CONV_TG_REG_M_SIZE; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W8(dst_trans,
                    8, k, buf_d, n_pad);
            }

            dst_trans += 8 * CONV_TG_REG_M_SIZE;
        }
        else if (n_pad - j == 4)
        {
            for (k = 0; k < CONV_TG_REG_M_SIZE; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W4(dst_trans,
                    4, k, buf_d, n_pad);
            }

            dst_trans += 4 * CONV_TG_REG_M_SIZE;
        }

        buf_d = buffer;
        for (j = 0; j < CONV_TG_REG_M_SIZE; j++)
        {
            for (k = 0; k < dst_trans_h; k++)
            {
                conv_tg_line_copy(buf_d + k * dst_trans_w,
                    dst_trans_w, dst + k * dst_w);
            }
            buf_d += n_pad;
            dst += dst_align;
        }
    }
    if (i < num_outs)
    {
        int m = num_outs - i;
        int m_pad = TG_PADDING(m, 4);

        for (j = 0; j < m; j++)
        {
            vb[j] = vdupq_n_f32(bias[i + j]);
        }

        float *buf_d = buffer;
        for (j = 0; j <= n_pad - CONV_TG_REG_N_SIZE;
            j += CONV_TG_REG_N_SIZE)
        {
            for (k = 0; k < m; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W12(dst_trans,
                    CONV_TG_REG_N_SIZE, k, buf_d, n_pad);
            }

            buf_d += CONV_TG_REG_N_SIZE;
            dst_trans += m_pad * CONV_TG_REG_N_SIZE;
        }
        if (n_pad - j == 8)
        {
            for (k = 0; k < m; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W8(dst_trans,
                    8, k, buf_d, n_pad);
            }
        }
        else if (n_pad - j == 4)
        {
            for (k = 0; k < m; k++)
            {
                CONV_TG_F3S1_DST_TRANS_W4(dst_trans,
                    4, k, buf_d, n_pad);
            }
        }

        buf_d = buffer;
        for (j = 0; j < m; j++)
        {
            for (k = 0; k < dst_trans_h; k++)
            {
                conv_tg_line_copy(buf_d + k * dst_trans_w,
                    dst_trans_w, dst + k * dst_w);
            }
            buf_d += n_pad;
            dst += dst_align;
        }
    }
}

static void conv_tg_f3s1_one_pad_wmax(float *src_pad,
    int src_pad_h,
    int channels,
    float *src_trans,
    float *filters_cvt,
    float *dst_trans,
    float *bias,
    int num_outs,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    int k = channels * 3 * 3;
    int k_pad = TG_PADDING(k, 4);

    int src_trans_h = src_pad_h - 3 + 1;

    float *src_trans_d = src_trans;

    for (i = 0; i <= channels - CONV_TG_F3_CHN_SIZE;
        i += CONV_TG_F3_CHN_SIZE)
    {
        conv_tg_f3s1_src_trans_wmax(src_pad,
            src_pad_h, CONV_TG_F3_CHN_SIZE, src_trans_d);

        src_pad += src_pad_h *
            CONV_TG_F3_CHN_SIZE * CONV_TG_F3S1_SRC_SIZE;
        src_trans_d += src_trans_h *
            CONV_TG_F3S1_K_SIZE * CONV_TG_F3S1_DST_SIZE;
    }
    if (i < channels)
    {
        conv_tg_f3s1_src_trans_wmax(src_pad,
            src_pad_h, channels - i, src_trans_d);
    }

    for (i = 0; i <= num_outs - CONV_TG_F3S1_M_SIZE;
        i += CONV_TG_F3S1_M_SIZE)
    {
        src_trans_d = src_trans;

        memset(dst_trans, 0, src_trans_h * CONV_TG_F3S1_M_SIZE *
            CONV_TG_F3S1_DST_SIZE * sizeof(float));

        for (j = 0; j <= k - CONV_TG_F3S1_K_SIZE;
            j += CONV_TG_F3S1_K_SIZE)
        {
            sgemm_kernel_fp32_m1(CONV_TG_F3S1_M_SIZE,
                src_trans_h * CONV_TG_F3S1_DST_SIZE,
                CONV_TG_F3S1_K_SIZE, filters_cvt,
                src_trans_d, dst_trans);

            filters_cvt += CONV_TG_F3S1_M_SIZE * CONV_TG_F3S1_K_SIZE;
            src_trans_d += CONV_TG_F3S1_K_SIZE *
                src_trans_h * CONV_TG_F3S1_DST_SIZE;
        }
        if (j < k)
        {
            sgemm_kernel_fp32_m1(CONV_TG_F3S1_M_SIZE,
                src_trans_h * CONV_TG_F3S1_DST_SIZE,
                k_pad - j, filters_cvt, src_trans_d,
                dst_trans);

            filters_cvt += (k_pad - j) * CONV_TG_F3S1_M_SIZE;
        }

        conv_tg_f3s1_dst_trans_wmax(dst_trans, src_trans_h,
            CONV_TG_F3S1_M_SIZE, bias + i, dst_h, dst_w, dst);

        dst += dst_h * dst_w * CONV_TG_F3S1_M_SIZE;
    }
    if (i < num_outs)
    {
        int m_pad = TG_PADDING(num_outs - i, 4);
        src_trans_d = src_trans;

        memset(dst_trans, 0, src_trans_h * m_pad *
            CONV_TG_F3S1_DST_SIZE * sizeof(float));

        for (j = 0; j <= k - CONV_TG_F3S1_K_SIZE;
            j += CONV_TG_F3S1_K_SIZE)
        {
            sgemm_kernel_fp32_m1(m_pad, src_trans_h *
                CONV_TG_F3S1_DST_SIZE, CONV_TG_F3S1_K_SIZE,
                filters_cvt, src_trans_d, dst_trans);

            filters_cvt += m_pad * CONV_TG_F3S1_K_SIZE;
            src_trans_d += CONV_TG_F3S1_K_SIZE *
                src_trans_h * CONV_TG_F3S1_DST_SIZE;
        }
        if (j < k)
        {
            sgemm_kernel_fp32_m1(m_pad, src_trans_h *
                CONV_TG_F3S1_DST_SIZE, k_pad - j,
                filters_cvt, src_trans_d, dst_trans);
        }

        conv_tg_f3s1_dst_trans_wmax(dst_trans, src_trans_h,
            num_outs - i, bias + i, dst_h, dst_w, dst);
    }
}

static void conv_tg_f3s1_one_pad(float *src_pad,
    int src_pad_h,
    int src_pad_w,
    int channels,
    float *src_trans,
    float *filters_cvt,
    float *dst_trans,
    float *bias,
    int num_outs,
    float *buffer,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    int k = channels * 3 * 3;
    int k_pad = TG_PADDING(k, 4);

    int src_trans_h = src_pad_h - 3 + 1;
    int src_trans_w = src_pad_w - 3 + 1;

    int n = src_trans_h * src_trans_w;
    int n_pad = TG_PADDING(n, 4);

    float *src_trans_d = src_trans;

    for (i = 0; i <= channels - CONV_TG_F3_CHN_SIZE;
        i += CONV_TG_F3_CHN_SIZE)
    {
        conv_tg_f3s1_src_trans(src_pad, src_pad_h,
            src_pad_w, CONV_TG_F3_CHN_SIZE, buffer,
            src_trans_d);

        src_pad += src_pad_h * src_pad_w *
            CONV_TG_F3_CHN_SIZE;
        src_trans_d += n_pad * CONV_TG_F3S1_K_SIZE;
    }
    if (i < channels)
    {
        conv_tg_f3s1_src_trans(src_pad, src_pad_h,
            src_pad_w, channels - i, buffer, src_trans_d);
    }

    for (i = 0; i <= num_outs - CONV_TG_F3S1_M_SIZE;
        i += CONV_TG_F3S1_M_SIZE)
    {
        src_trans_d = src_trans;

        memset(dst_trans, 0,
            n_pad * CONV_TG_F3S1_M_SIZE * sizeof(float));

        for (j = 0; j <= k - CONV_TG_F3S1_K_SIZE;
            j += CONV_TG_F3S1_K_SIZE)
        {
            sgemm_kernel_fp32_m1(CONV_TG_F3S1_M_SIZE, n_pad,
                CONV_TG_F3S1_K_SIZE, filters_cvt,
                src_trans_d, dst_trans);

            filters_cvt += CONV_TG_F3S1_M_SIZE *
                CONV_TG_F3S1_K_SIZE;
            src_trans_d += n_pad * CONV_TG_F3S1_K_SIZE;
        }
        if (j < k)
        {
            sgemm_kernel_fp32_m1(CONV_TG_F3S1_M_SIZE, n_pad,
                k_pad - j, filters_cvt, src_trans_d,
                dst_trans);

            filters_cvt += (k_pad - j) * CONV_TG_F3S1_M_SIZE;
        }

        conv_tg_f3s1_dst_trans(dst_trans, src_trans_h,
            src_trans_w, CONV_TG_F3S1_M_SIZE, bias + i,
            buffer, dst_h, dst_w, dst);

        dst += dst_h * dst_w * CONV_TG_F3S1_M_SIZE;
    }
    if (i < num_outs)
    {
        int m_pad = TG_PADDING(num_outs - i, 4);
        src_trans_d = src_trans;

        memset(dst_trans, 0, n_pad * m_pad * sizeof(float));

        for (j = 0; j <= k - CONV_TG_F3S1_K_SIZE;
            j += CONV_TG_F3S1_K_SIZE)
        {
            sgemm_kernel_fp32_m1(m_pad, n_pad, CONV_TG_F3S1_K_SIZE,
                filters_cvt, src_trans_d, dst_trans);

            filters_cvt += m_pad * CONV_TG_F3S1_K_SIZE;
            src_trans_d += n_pad * CONV_TG_F3S1_K_SIZE;
        }
        if (j < k)
        {
            sgemm_kernel_fp32_m1(m_pad, n_pad, k_pad - j,
                filters_cvt, src_trans_d, dst_trans);
        }

        conv_tg_f3s1_dst_trans(dst_trans, src_trans_h,
            src_trans_w, num_outs - i, bias + i, buffer,
            dst_h, dst_w, dst);
    }
}

void conv_tile_gemm_f3s1_m1(float *src,
    int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    float *filters_cvt,
    float *bias,
    int num_outs,
    void *temp_buffer,
    float *dst)
{
    int i, j;

    int src_pad_h = src_h + 2 * padding_h;
    int src_pad_w = src_w + 2 * padding_w;

    int dst_pad_h = src_pad_h - 3 + 1;
    int dst_pad_w = src_pad_w - 3 + 1;

    int k = 3 * 3 * channels;
    int k_pad = TG_PADDING(k, 4);
    int m_pad = TG_PADDING(num_outs, 4);

    int buffer_len =
        MAX_INT(CONV_TG_REG_M_SIZE * CONV_TG_F3S1_N_SIZE,
            3 * 3 * CONV_TG_F3S1_N_SIZE);
    int src_trans_len = k_pad * CONV_TG_F3S1_N_SIZE;

    float *buffer = (float*)temp_buffer;
    float *src_trans = buffer + buffer_len;
    float *dst_trans = src_trans + src_trans_len;
    float *src_pad = dst_trans;

    int hori_beg = -padding_w;
    int hori_end = hori_beg + src_pad_w;

    int vert_beg = -padding_h;
    int vert_end = vert_beg + src_pad_h;

    int h_idx = vert_beg;
    int w_idx = hori_beg;

    int real_src_h = 0, real_src_w = 0;
    int h_dst_idx = 0, w_dst_idx = 0;

    while (conv_tg_blocking_image(src, src_h, src_w, channels,
        hori_beg, hori_end, vert_beg, vert_end, h_idx, w_idx,
        src_pad, real_src_h, real_src_w))
    {
        int real_dst_h = real_src_h - 3 + 1;
        int real_dst_w = real_src_w - 3 + 1;

        if (real_src_w == CONV_TG_F3S1_SRC_SIZE)
        {
            conv_tg_f3s1_one_pad_wmax(src_pad, real_src_h,
                channels, src_trans, filters_cvt, dst_trans,
                bias, num_outs, dst_pad_h, dst_pad_w,
                dst + h_dst_idx * dst_pad_w + w_dst_idx);
        }
        else
        {
            conv_tg_f3s1_one_pad(src_pad, real_src_h,
                real_src_w, channels, src_trans, filters_cvt,
                dst_trans, bias, num_outs, buffer, dst_pad_h,
                dst_pad_w, dst + h_dst_idx * dst_pad_w + w_dst_idx);
        }

        w_dst_idx += real_dst_w;
        if (w_dst_idx >= dst_pad_w)
        {
            w_dst_idx = 0;
            h_dst_idx += real_dst_h;
        }
    }
}

