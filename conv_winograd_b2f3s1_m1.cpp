#include "conv_winograd_b2f3s1_m1.h"
#include "conv_winograd_b2f3s1_params_m1.h"

#include <cstring>
#include <arm_neon.h>

#define MAX_INT(A, B) ((A) < (B) ? (B) : (A))
#define MIN_INT(A, B) ((A) < (B) ? (A) : (B))
#define WINO_PADDING(SIZE, ALIGN) (((SIZE) + (ALIGN) - 1) / (ALIGN) * (ALIGN))
#define CONV_TRANSPOSE_4X4(Q0, Q1, Q2, Q3) \
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

extern void sgemm_kernel_fp32_m1(int,
    int,
    int,
    float *,
    float *,
    float *);

int conv_winograd_b2f3s1_buf_size_m1(int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    int num_outs)
{
    int src_pad_h = src_h + 2 * padding_h;
    int src_pad_w = src_w + 2 * padding_w;

    int dst_h = src_pad_h - 3 + 1;
    int dst_w = src_pad_w - 3 + 1;

    int dst_pad_h = WINO_PADDING(dst_h, 2);
    int dst_pad_w = WINO_PADDING(dst_w, 2);

    src_pad_h = dst_pad_h + 3 - 1;
    src_pad_w = dst_pad_w + 3 - 1;

    int src_pad_size, src_trans_size, dst_trans_size;
    int total_size = 0;

    int buffer_size = 4 * CONV_WGB2_F3S1_N_SIZE *
        sizeof(float);

    int fcvt_size = (16 * 4 + 16 * 4 + 16 * 4 * 8) * sizeof(float);

    int m = MIN_INT(CONV_WGB2_F3S1_M_SIZE, WINO_PADDING(num_outs,4));

    if(dst_w == CONV_WG_DST_PAD_SIZE &&
        dst_h < CONV_WG_DST_PAD_SIZE)
    {
        src_pad_size = channels * src_pad_h *
            CONV_WG_SRC_PAD_SIZE * sizeof(float);
        int n = dst_pad_h * CONV_WG_TRANS_SIZE / 2;
        src_trans_size = 16 * n *
            WINO_PADDING(channels, 4) * sizeof(float);
        dst_trans_size = 16 * m * n * sizeof(float);
    }
    else if(dst_w < CONV_WG_DST_PAD_SIZE &&
        dst_h < CONV_WG_DST_PAD_SIZE)
    {
        src_pad_size = channels * src_pad_h *
            src_pad_w * sizeof(float);
        int n = WINO_PADDING(dst_pad_h * dst_pad_w / 4, 4);
        src_trans_size = 16 * n *
            WINO_PADDING(channels, 4) * sizeof(float);
        dst_trans_size = 16 * m * n * sizeof(float);
    }
    else
    {
        src_pad_size = channels * CONV_WG_SRC_PAD_SIZE *
            CONV_WG_SRC_PAD_SIZE * sizeof(float);
        src_trans_size = 16 * CONV_WGB2_F3S1_N_SIZE *
            WINO_PADDING(channels, 4) * sizeof(float);
        dst_trans_size = 16 * m *
            CONV_WGB2_F3S1_N_SIZE * sizeof(float);
    }

    total_size += src_trans_size + buffer_size +
        MAX_INT(src_pad_size, dst_trans_size);

    total_size = MAX_INT(total_size, fcvt_size);

    return total_size;
}

int conv_winograd_b2f3s1_cvt_filter_size_m1(int channels,
    int num_outs)
{
    int m = WINO_PADDING(num_outs, 4);
    int k = WINO_PADDING(channels, 4);
    return 16 * m * k * sizeof(float);
}

#define CONV_WGB2_F3S1_CVTFLT_INIT(SRC, DST) \
{ \
    vr[0] = vld1q_f32((SRC) + 0 * 9 + 0); \
    vr[1] = vld1q_f32((SRC) + 1 * 9 + 0); \
    vr[2] = vld1q_f32((SRC) + 2 * 9 + 0); \
    vr[3] = vld1q_f32((SRC) + 3 * 9 + 0); \
    CONV_TRANSPOSE_4X4(vr[0], vr[1], vr[2], vr[3]); \
    vst1q_f32((DST) + 0 * 4, vr[0]); \
    vst1q_f32((DST) + 1 * 4, vr[1]); \
    vst1q_f32((DST) + 2 * 4, vr[2]); \
    vst1q_f32((DST) + 3 * 4, vr[3]); \
    vr[0] = vld1q_f32((SRC) + 0 * 9 + 4); \
    vr[1] = vld1q_f32((SRC) + 1 * 9 + 4); \
    vr[2] = vld1q_f32((SRC) + 2 * 9 + 4); \
    vr[3] = vld1q_f32((SRC) + 3 * 9 + 4); \
    CONV_TRANSPOSE_4X4(vr[0], vr[1], vr[2], vr[3]); \
    vst1q_f32((DST) + 4 * 4, vr[0]); \
    vst1q_f32((DST) + 5 * 4, vr[1]); \
    vst1q_f32((DST) + 6 * 4, vr[2]); \
    vst1q_f32((DST) + 7 * 4, vr[3]); \
    (DST)[8 * 4 + 0] = (SRC)[0 * 9 + 8]; \
    (DST)[8 * 4 + 1] = (SRC)[1 * 9 + 8]; \
    (DST)[8 * 4 + 2] = (SRC)[2 * 9 + 8]; \
    (DST)[8 * 4 + 3] = (SRC)[3 * 9 + 8]; \
}

#define CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(SRC, DST) \
{ \
    vr[12] = vld1q_f32((SRC) + 0 * 4); \
    vr[13] = vld1q_f32((SRC) + 1 * 4); \
    vr[14] = vld1q_f32((SRC) + 2 * 4); \
    vr[0] = (float32x4_t)veorq_u32((uint32x4_t)vr[0], (uint32x4_t)vr[0]); \
    vr[1] = (float32x4_t)veorq_u32((uint32x4_t)vr[1], (uint32x4_t)vr[1]); \
    vr[2] = (float32x4_t)veorq_u32((uint32x4_t)vr[2], (uint32x4_t)vr[2]); \
    vr[3] = (float32x4_t)veorq_u32((uint32x4_t)vr[3], (uint32x4_t)vr[3]); \
    vr[4] = (float32x4_t)veorq_u32((uint32x4_t)vr[4], (uint32x4_t)vr[4]); \
    vr[5] = (float32x4_t)veorq_u32((uint32x4_t)vr[5], (uint32x4_t)vr[5]); \
    vr[6] = (float32x4_t)veorq_u32((uint32x4_t)vr[6], (uint32x4_t)vr[6]); \
    vr[7] = (float32x4_t)veorq_u32((uint32x4_t)vr[7], (uint32x4_t)vr[7]); \
    vr[8] = (float32x4_t)veorq_u32((uint32x4_t)vr[8], (uint32x4_t)vr[8]); \
    vr[9] = (float32x4_t)veorq_u32((uint32x4_t)vr[9], (uint32x4_t)vr[9]); \
    vr[10] = (float32x4_t)veorq_u32((uint32x4_t)vr[10], (uint32x4_t)vr[10]); \
    vr[11] = (float32x4_t)veorq_u32((uint32x4_t)vr[11], (uint32x4_t)vr[11]); \
    vr[0] = vmlaq_lane_f32(vr[0], vr[12], vg, 0); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[13], vg, 0); \
    vr[2] = vmlaq_lane_f32(vr[2], vr[14], vg, 0); \
    vr[3] = vmlaq_lane_f32(vr[3], vr[12], vg, 1); \
    vr[4] = vmlaq_lane_f32(vr[4], vr[13], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[14], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[12], vg, 1); \
    vr[7] = vmlaq_lane_f32(vr[7], vr[13], vg, 1); \
    vr[8] = vmlaq_lane_f32(vr[8], vr[14], vg, 1); \
    vr[12] = vld1q_f32((SRC) + 3 * 4); \
    vr[13] = vld1q_f32((SRC) + 4 * 4); \
    vr[14] = vld1q_f32((SRC) + 5 * 4); \
    vr[3] = vmlaq_lane_f32(vr[3], vr[12], vg, 1); \
    vr[4] = vmlaq_lane_f32(vr[4], vr[13], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[14], vg, 1); \
    vr[6] = vmlsq_lane_f32(vr[6], vr[12], vg, 1); \
    vr[7] = vmlsq_lane_f32(vr[7], vr[13], vg, 1); \
    vr[8] = vmlsq_lane_f32(vr[8], vr[14], vg, 1); \
    vr[12] = vld1q_f32((SRC) + 6 * 4); \
    vr[13] = vld1q_f32((SRC) + 7 * 4); \
    vr[14] = vld1q_f32((SRC) + 8 * 4); \
    vr[3] = vmlaq_lane_f32(vr[3], vr[12], vg, 1); \
    vr[4] = vmlaq_lane_f32(vr[4], vr[13], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[14], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[12], vg, 1); \
    vr[7] = vmlaq_lane_f32(vr[7], vr[13], vg, 1); \
    vr[8] = vmlaq_lane_f32(vr[8], vr[14], vg, 1); \
    vr[9] = vmlaq_lane_f32(vr[9], vr[12], vg, 0); \
    vr[10] = vmlaq_lane_f32(vr[10], vr[13], vg, 0); \
    vr[11] = vmlaq_lane_f32(vr[11], vr[14], vg, 0); \
    vst1q_f32((DST) + 0 * 4, vr[0]); \
    vst1q_f32((DST) + 1 * 4, vr[1]); \
    vst1q_f32((DST) + 2 * 4, vr[2]); \
    vst1q_f32((DST) + 3 * 4, vr[3]); \
    vst1q_f32((DST) + 4 * 4, vr[4]); \
    vst1q_f32((DST) + 5 * 4, vr[5]); \
    vst1q_f32((DST) + 6 * 4, vr[6]); \
    vst1q_f32((DST) + 7 * 4, vr[7]); \
    vst1q_f32((DST) + 8 * 4, vr[8]); \
    vst1q_f32((DST) + 9 * 4, vr[9]); \
    vst1q_f32((DST) + 10 * 4, vr[10]); \
    vst1q_f32((DST) + 11 * 4, vr[11]); \
}

#define CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(SRC, DST, LDD) \
{ \
    vr[8] = vld1q_f32((SRC) + 0 * 12 + 0); \
    vr[9] = vld1q_f32((SRC) + 1 * 12 + 0); \
    vr[0] = (float32x4_t)veorq_u32((uint32x4_t)vr[0], (uint32x4_t)vr[0]); \
    vr[1] = (float32x4_t)veorq_u32((uint32x4_t)vr[1], (uint32x4_t)vr[1]); \
    vr[2] = (float32x4_t)veorq_u32((uint32x4_t)vr[2], (uint32x4_t)vr[2]); \
    vr[3] = (float32x4_t)veorq_u32((uint32x4_t)vr[3], (uint32x4_t)vr[3]); \
    vr[4] = (float32x4_t)veorq_u32((uint32x4_t)vr[4], (uint32x4_t)vr[4]); \
    vr[5] = (float32x4_t)veorq_u32((uint32x4_t)vr[5], (uint32x4_t)vr[5]); \
    vr[6] = (float32x4_t)veorq_u32((uint32x4_t)vr[6], (uint32x4_t)vr[6]); \
    vr[7] = (float32x4_t)veorq_u32((uint32x4_t)vr[7], (uint32x4_t)vr[7]); \
    vr[0] = vmlaq_lane_f32(vr[0], vr[8], vg, 0); \
    vr[4] = vmlaq_lane_f32(vr[4], vr[9], vg, 0); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlaq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[8] = vld1q_f32((SRC) + 0 * 12 + 4); \
    vr[9] = vld1q_f32((SRC) + 1 * 12 + 4); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlsq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlsq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[8] = vld1q_f32((SRC) + 0 * 12 + 8); \
    vr[9] = vld1q_f32((SRC) + 1 * 12 + 8); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlaq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[3] = vmlaq_lane_f32(vr[3], vr[8], vg, 0); \
    vr[7] = vmlaq_lane_f32(vr[7], vr[9], vg, 0); \
    vst1q_f32((DST) + 0 * (LDD), vr[0]); \
    vst1q_f32((DST) + 1 * (LDD), vr[1]); \
    vst1q_f32((DST) + 2 * (LDD), vr[2]); \
    vst1q_f32((DST) + 3 * (LDD), vr[3]); \
    vst1q_f32((DST) + 4 * (LDD), vr[4]); \
    vst1q_f32((DST) + 5 * (LDD), vr[5]); \
    vst1q_f32((DST) + 6 * (LDD), vr[6]); \
    vst1q_f32((DST) + 7 * (LDD), vr[7]); \
    vr[8] = vld1q_f32((SRC) + 2 * 12 + 0); \
    vr[9] = vld1q_f32((SRC) + 3 * 12 + 0); \
    vr[0] = (float32x4_t)veorq_u32((uint32x4_t)vr[0], (uint32x4_t)vr[0]); \
    vr[1] = (float32x4_t)veorq_u32((uint32x4_t)vr[1], (uint32x4_t)vr[1]); \
    vr[2] = (float32x4_t)veorq_u32((uint32x4_t)vr[2], (uint32x4_t)vr[2]); \
    vr[3] = (float32x4_t)veorq_u32((uint32x4_t)vr[3], (uint32x4_t)vr[3]); \
    vr[4] = (float32x4_t)veorq_u32((uint32x4_t)vr[4], (uint32x4_t)vr[4]); \
    vr[5] = (float32x4_t)veorq_u32((uint32x4_t)vr[5], (uint32x4_t)vr[5]); \
    vr[6] = (float32x4_t)veorq_u32((uint32x4_t)vr[6], (uint32x4_t)vr[6]); \
    vr[7] = (float32x4_t)veorq_u32((uint32x4_t)vr[7], (uint32x4_t)vr[7]); \
    vr[0] = vmlaq_lane_f32(vr[0], vr[8], vg, 0); \
    vr[4] = vmlaq_lane_f32(vr[4], vr[9], vg, 0); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlaq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[8] = vld1q_f32((SRC) + 2 * 12 + 4); \
    vr[9] = vld1q_f32((SRC) + 3 * 12 + 4); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlsq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlsq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[8] = vld1q_f32((SRC) + 2 * 12 + 8); \
    vr[9] = vld1q_f32((SRC) + 3 * 12 + 8); \
    vr[1] = vmlaq_lane_f32(vr[1], vr[8], vg, 1); \
    vr[5] = vmlaq_lane_f32(vr[5], vr[9], vg, 1); \
    vr[2] = vmlaq_lane_f32(vr[2], vr[8], vg, 1); \
    vr[6] = vmlaq_lane_f32(vr[6], vr[9], vg, 1); \
    vr[3] = vmlaq_lane_f32(vr[3], vr[8], vg, 0); \
    vr[7] = vmlaq_lane_f32(vr[7], vr[9], vg, 0); \
    vst1q_f32((DST) + 8 * (LDD), vr[0]); \
    vst1q_f32((DST) + 9 * (LDD), vr[1]); \
    vst1q_f32((DST) + 10 * (LDD), vr[2]); \
    vst1q_f32((DST) + 11 * (LDD), vr[3]); \
    vst1q_f32((DST) + 12 * (LDD), vr[4]); \
    vst1q_f32((DST) + 13 * (LDD), vr[5]); \
    vst1q_f32((DST) + 14 * (LDD), vr[6]); \
    vst1q_f32((DST) + 15 * (LDD), vr[7]); \
}

static void conv_wg_flt_one_blk(float *cvt_filter,
    int stride,
    int channels,
    int num_outs,
    float *filter_cvt)
{
    int i, j, ii, jj;

    float32x4_t vr[8];
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (i = 0; i <= num_outs - 8; i += 8)
    {
        for (j = 0; j <= channels - 4; j += 4)
        {
            float *cfd = cvt_filter +
                i * stride + j;

            vr[0] = vld1q_f32(cfd + 0 * stride);
            vr[1] = vld1q_f32(cfd + 1 * stride);
            vr[2] = vld1q_f32(cfd + 2 * stride);
            vr[3] = vld1q_f32(cfd + 3 * stride);
            vr[4] = vld1q_f32(cfd + 4 * stride);
            vr[5] = vld1q_f32(cfd + 5 * stride);
            vr[6] = vld1q_f32(cfd + 6 * stride);
            vr[7] = vld1q_f32(cfd + 7 * stride);
            CONV_TRANSPOSE_4X4(vr[0], vr[1], vr[2], vr[3]);
            CONV_TRANSPOSE_4X4(vr[4], vr[5], vr[6], vr[7]);
            vst1q_f32(filter_cvt + 0, vr[0]);
            vst1q_f32(filter_cvt + 4, vr[4]);
            vst1q_f32(filter_cvt + 8, vr[1]);
            vst1q_f32(filter_cvt + 12, vr[5]);
            vst1q_f32(filter_cvt + 16, vr[2]);
            vst1q_f32(filter_cvt + 20, vr[6]);
            vst1q_f32(filter_cvt + 24, vr[3]);
            vst1q_f32(filter_cvt + 28, vr[7]);

            filter_cvt += 32;
        }
        if (j < channels)
        {
            float *cfd = cvt_filter +
                i * stride + j;

            for (ii = 0; ii < channels - j; ii++)
            {
                for (jj = 0; jj < 8; jj++)
                {
                    filter_cvt[jj] = cfd[jj * stride + ii];
                }
                filter_cvt += 8;
            }
            for (; ii < 4; ii++)
            {
                vst1q_f32(filter_cvt + 0, vzero);
                vst1q_f32(filter_cvt + 4, vzero);
                filter_cvt += 8;
            }
        }
    }
    if (num_outs - i > 4)
    {
        int k = WINO_PADDING(channels, 4);

        for (j = 0; j < channels; j++)
        {
            float *cfd = cvt_filter +
                i * stride + j;

            for (ii = 0; ii < num_outs - i; ii++)
            {
                filter_cvt[ii] = cfd[ii * stride];
            }
            for (; ii < 8; ii++)
            {
                filter_cvt[ii] = 0.0f;
            }
            filter_cvt += 8;
        }
        for (; j < k; j++)
        {
            vst1q_f32(filter_cvt + 0, vzero);
            vst1q_f32(filter_cvt + 4, vzero);
            filter_cvt += 8;
        }
    }
    else if (i < num_outs)
    {
        int k = WINO_PADDING(channels, 4);

        for (j = 0; j < channels; j++)
        {
            float *cfd = cvt_filter +
                i * stride + j;

            for (ii = 0; ii < num_outs - i; ii++)
            {
                filter_cvt[ii] = cfd[ii * stride];
            }
            for (; ii < 4; ii++)
            {
                filter_cvt[ii] = 0.0f;
            }
            filter_cvt += 4;
        }
        for (; j < k; j++)
        {
            vst1q_f32(filter_cvt + 0, vzero);
            filter_cvt += 4;
        }
    }
}

static void conv_wgb2_f3s1_flt_trans_one_blk(float *filters,
    int channels,
    int k_blk,
    int m_blk,
    float *ping_buf,
    float *pong_buf,
    float *cvt_filter,
    float *filter_cvt)
{
    int ii, jj, kk, k;

    int flt_align_blk = WINO_PADDING(m_blk, 4)
        * WINO_PADDING(k_blk, 4);
    int flt_align = 8 * 4;

    static float g_vals[2] = {1.0f, 0.5f};

    float32x4_t vr[15];
    float32x2_t vg = vld1_f32(g_vals);

    for(ii = 0; ii <= m_blk - 8; ii += 8)
    {
        for(jj = 0; jj <= k_blk - 4; jj += 4)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < 8; ++i)
            {
                CONV_WGB2_F3S1_CVTFLT_INIT(filters_d,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    cvt_filter_d, flt_align);

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, 4, 8,
                    filter_cvt + kk * flt_align_blk);
            }

            filter_cvt += flt_align;
        }
        if(jj < k_blk)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < 8; ++i)
            {
                memcpy(pong_buf, filters_d, (k_blk - jj) * 9 * sizeof(float));
                CONV_WGB2_F3S1_CVTFLT_INIT(pong_buf,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    ping_buf, 4);

                for(kk = 0; kk < 16; ++kk)
                {
                    float *cfd = cvt_filter_d + kk * flt_align;

                    for(k = 0; k < k_blk - jj; ++k)
                    {
                        cfd[k] = ping_buf[kk * 4 + k];
                    }
                }

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, k_blk - jj,
                    8, filter_cvt + kk * flt_align_blk);
            }

            filter_cvt += flt_align;
        }
    }
    if(m_blk - ii > 4)
    {
        for(jj = 0; jj <= k_blk - 4; jj += 4)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < (m_blk - ii); ++i)
            {
                CONV_WGB2_F3S1_CVTFLT_INIT(filters_d,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    cvt_filter_d, flt_align);

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, 4, m_blk - ii,
                    filter_cvt + kk * flt_align_blk);
            }

            filter_cvt += flt_align;
        }
        if(jj < k_blk)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < (m_blk - ii); ++i)
            {
                memcpy(pong_buf, filters_d, (k_blk - jj) * 9 * sizeof(float));
                CONV_WGB2_F3S1_CVTFLT_INIT(pong_buf,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    ping_buf, 4);

                for(kk = 0; kk < 16; ++kk)
                {
                    float *cfd = cvt_filter_d + kk * flt_align;

                    for(k = 0; k < k_blk - jj; ++k)
                    {
                        cfd[k] = ping_buf[kk * 4 + k];
                    }
                }

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, k_blk - jj,
                    m_blk - ii, filter_cvt + kk * flt_align_blk);
            }

            filter_cvt += flt_align;
        }
    }
    else if(ii < m_blk)
    {
        for(jj = 0; jj <= k_blk - 4; jj += 4)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < (m_blk - ii); ++i)
            {
                CONV_WGB2_F3S1_CVTFLT_INIT(filters_d,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    cvt_filter_d, flt_align);

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, 4, m_blk - ii,
                    filter_cvt + kk * flt_align_blk);
            }

            filter_cvt += 4 * 4;
        }
        if(jj < k_blk)
        {
            float *filters_d = filters + (ii * channels + jj) * 3 * 3;
            float *cvt_filter_d = cvt_filter;

            for(int i = 0; i < (m_blk - ii); ++i)
            {
                memcpy(pong_buf, filters_d, (k_blk - jj) * 9 * sizeof(float));
                CONV_WGB2_F3S1_CVTFLT_INIT(pong_buf,
                    ping_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_LEFT(ping_buf,
                    pong_buf);
                CONV_WGB2_F3S1_CVTFLT_MUL_RIGHT(pong_buf,
                    ping_buf, 4);

                for(kk = 0; kk < 16; ++kk)
                {
                    float *cfd = cvt_filter_d + kk * flt_align;

                    for(k = 0; k < k_blk - jj; ++k)
                    {
                        cfd[k] = ping_buf[kk * 4 + k];
                    }
                }

                filters_d += channels * 3 * 3;
                cvt_filter_d += 4;
            }

            for(kk = 0; kk < 16; ++kk)
            {
                float *cfd = cvt_filter + kk * flt_align;

                conv_wg_flt_one_blk(cfd, 4, k_blk - jj,
                    m_blk - ii, filter_cvt + kk * flt_align_blk);
            }
        }
    }
}

static void conv_wgb2_f3s1_blocking_cvt_filter(float *filters,
    int channels,
    int num_outs,
    int m_blk,
    int k_blk,
    float *ping_buf,
    float *pong_buf,
    float *cvt_filter,
    float *filter_cvt)
{
    int i, j;

    for(i = 0; i <= num_outs - m_blk; i += m_blk)
    {
        for(j = 0; j <= channels - k_blk; j += k_blk)
        {
            float *filters_d = filters + (i * channels + j) * 3 * 3;

            conv_wgb2_f3s1_flt_trans_one_blk(filters_d, channels, k_blk,
                m_blk, ping_buf, pong_buf, cvt_filter, filter_cvt);

            filter_cvt += 16 * m_blk * k_blk;
        }
        if(j < channels)
        {
            float *filters_d = filters + (i * channels + j) * 3 * 3;

            int k = WINO_PADDING(channels - j, 4);

            conv_wgb2_f3s1_flt_trans_one_blk(filters_d, channels, channels - j,
                m_blk, ping_buf, pong_buf, cvt_filter, filter_cvt);

            filter_cvt += 16 * m_blk * k;
        }
    }
    if(i < num_outs)
    {
        int m = WINO_PADDING(num_outs - i, 4);

        for(j = 0; j <= channels - k_blk; j += k_blk)
        {
            float *filters_d = filters + (i * channels + j) * 3 * 3;

            conv_wgb2_f3s1_flt_trans_one_blk(filters_d, channels,
                k_blk, num_outs - i, ping_buf, pong_buf, cvt_filter, filter_cvt);

            filter_cvt += 16 * m * k_blk;
        }
        if(j < channels)
        {
            float *filters_d = filters + (i * channels + j) * 3 * 3;

            conv_wgb2_f3s1_flt_trans_one_blk(filters_d, channels, channels - j,
                num_outs - i, ping_buf, pong_buf, cvt_filter, filter_cvt);
        }
    }
}

void conv_winograd_b2f3s1_cvt_filter_m1(float *filters,
    int channels,
    int num_outs,
    void *temp_buffer,
    float *cvt_filters)
{
    float *ping_buf = (float*)temp_buffer;
    float *pong_buf = ping_buf + 16 * 4;
    float *cvt_filter = pong_buf + 16 * 4;

    conv_wgb2_f3s1_blocking_cvt_filter(filters, channels, num_outs,
        CONV_WGB2_F3S1_M_SIZE, CONV_WGB2_F3S1_K_SIZE, ping_buf,
        pong_buf, cvt_filter, cvt_filters);
}

static void conv_wg_padding_image(float *src,
    int src_h,
    int src_w,
    int channels,
    int left,
    int right,
    int top,
    int bottom,
    float *src_pad)
{
    int i, j, k;

    int src_pad_w = src_w + left + right;

    for (i = 0; i < channels; i++)
    {
        memset(src_pad, 0, top * src_pad_w * sizeof(float));
        src_pad += top * src_pad_w;

        for (j = 0; j < src_h; j++)
        {
            for (k = 0; k < left; k++)
            {
                src_pad[k] = 0.0f;
            }

            memcpy(src_pad + left, src, src_w * sizeof(float));
            src += src_w;

            for (k = left + src_w; k < src_pad_w; k++)
            {
                src_pad[k] = 0.0f;
            }
            src_pad += src_pad_w;
        }

        memset(src_pad, 0, bottom * src_pad_w * sizeof(float));
        src_pad += bottom * src_pad_w;
    }
}

#define CONV_WGB2_F3S1_SRC_TRANS_WMAX(SRC, LDS, DST, LDD) \
{ \
    float32x4x2_t vi[4]; \
    float32x4_t vr[8]; \
    vi[0] = vld2q_f32((SRC) + 0 * (LDS) + 0); \
    vi[1] = vld2q_f32((SRC) + 0 * (LDS) + 2); \
    vi[2] = vld2q_f32((SRC) + 2 * (LDS) + 0); \
    vi[3] = vld2q_f32((SRC) + 2 * (LDS) + 2); \
    vr[0] = vsubq_f32(vi[0].val[0], vi[2].val[0]); \
    vr[1] = vsubq_f32(vi[0].val[1], vi[2].val[1]); \
    vr[2] = vsubq_f32(vi[1].val[0], vi[3].val[0]); \
    vr[3] = vsubq_f32(vi[1].val[1], vi[3].val[1]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 0 * (LDD), vr[4]); \
    vst1q_f32((DST) + 1 * (LDD), vr[5]); \
    vst1q_f32((DST) + 2 * (LDD), vr[6]); \
    vst1q_f32((DST) + 3 * (LDD), vr[7]); \
    vi[0] = vld2q_f32((SRC) + 1 * (LDS) + 0); \
    vi[1] = vld2q_f32((SRC) + 1 * (LDS) + 2); \
    vr[0] = vaddq_f32(vi[0].val[0], vi[2].val[0]); \
    vr[1] = vaddq_f32(vi[0].val[1], vi[2].val[1]); \
    vr[2] = vaddq_f32(vi[1].val[0], vi[3].val[0]); \
    vr[3] = vaddq_f32(vi[1].val[1], vi[3].val[1]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 4 * (LDD), vr[4]); \
    vst1q_f32((DST) + 5 * (LDD), vr[5]); \
    vst1q_f32((DST) + 6 * (LDD), vr[6]); \
    vst1q_f32((DST) + 7 * (LDD), vr[7]); \
    vr[0] = vsubq_f32(vi[2].val[0], vi[0].val[0]); \
    vr[1] = vsubq_f32(vi[2].val[1], vi[0].val[1]); \
    vr[2] = vsubq_f32(vi[3].val[0], vi[1].val[0]); \
    vr[3] = vsubq_f32(vi[3].val[1], vi[1].val[1]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 8 * (LDD), vr[4]); \
    vst1q_f32((DST) + 9 * (LDD), vr[5]); \
    vst1q_f32((DST) + 10 * (LDD), vr[6]); \
    vst1q_f32((DST) + 11 * (LDD), vr[7]); \
    vi[2] = vld2q_f32((SRC) + 3 * (LDS) + 0); \
    vi[3] = vld2q_f32((SRC) + 3 * (LDS) + 2); \
    vr[0] = vsubq_f32(vi[0].val[0], vi[2].val[0]); \
    vr[1] = vsubq_f32(vi[0].val[1], vi[2].val[1]); \
    vr[2] = vsubq_f32(vi[1].val[0], vi[3].val[0]); \
    vr[3] = vsubq_f32(vi[1].val[1], vi[3].val[1]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 12 * (LDD), vr[4]); \
    vst1q_f32((DST) + 13 * (LDD), vr[5]); \
    vst1q_f32((DST) + 14 * (LDD), vr[6]); \
    vst1q_f32((DST) + 15 * (LDD), vr[7]); \
}

static void conv_wgb2_f3s1_src_trans_col_wmax(float *src_pad,
    int src_pad_h,
    int channels,
    float *src_trans)
{

    int k = WINO_PADDING(channels, 4);

    int trans_align = CONV_WG_TRANS_SIZE * k *
        (src_pad_h - 3 + 1) / 2;

    int i,j;
    for (i = 0; i < channels; i++)
    {
        CONV_WGB2_F3S1_SRC_TRANS_WMAX(
            src_pad +  0, CONV_WG_SRC_PAD_SIZE,
            src_trans + 0, trans_align);
        CONV_WGB2_F3S1_SRC_TRANS_WMAX(
            src_pad +  8, CONV_WG_SRC_PAD_SIZE,
            src_trans + 4, trans_align);
        CONV_WGB2_F3S1_SRC_TRANS_WMAX(
            src_pad + 16, CONV_WG_SRC_PAD_SIZE,
            src_trans + 8, trans_align);

        src_pad += src_pad_h * CONV_WG_SRC_PAD_SIZE;
        src_trans += CONV_WG_TRANS_SIZE;
    }
    if (i < k)
    {
        for (j = 0; j < 16; j++)
        {
            memset(src_trans + j * trans_align, 0,
                (k - i) * CONV_WG_TRANS_SIZE * sizeof(float));
        }
    }
}

static inline void conv_wgb2_f3s1_src_trans_hwmax(float *src_pad,
    int channels,
    float *src_trans)
{
    int i;

    int k = WINO_PADDING(channels, 4);


    for (i = 0; i < CONV_WG_TRANS_SIZE; i++)
    {
        conv_wgb2_f3s1_src_trans_col_wmax(src_pad,
            CONV_WG_SRC_PAD_SIZE, channels, src_trans);

        src_pad += 2 * CONV_WG_SRC_PAD_SIZE;
        src_trans += k * CONV_WG_TRANS_SIZE;
    }
}

static inline void conv_wgb2_f3s1_src_trans_wmax(float *src_pad,
    int src_pad_h,
    int channels,
    float *src_trans)
{
    int i;

    int k = WINO_PADDING(channels, 4);

    for (i = 0; i <= src_pad_h - 4; i += 2)
    {
        conv_wgb2_f3s1_src_trans_col_wmax(src_pad,
            src_pad_h, channels, src_trans);

        src_pad += 2 * CONV_WG_SRC_PAD_SIZE;
        src_trans += k * CONV_WG_TRANS_SIZE;
    }
}

#define CONV_WGB2_F3S1_SRC_TRANS(SRC0, SRC1, SRC2, SRC3, LDS, DST, LDD) \
{ \
    float32x4_t vi[8]; \
    float32x4_t vr[8]; \
    vi[0] = vld1q_f32((SRC0) + 0 * (LDS)); \
    vi[1] = vld1q_f32((SRC1) + 0 * (LDS)); \
    vi[2] = vld1q_f32((SRC2) + 0 * (LDS)); \
    vi[3] = vld1q_f32((SRC3) + 0 * (LDS)); \
    CONV_TRANSPOSE_4X4(vi[0], vi[1], vi[2], vi[3]); \
    vi[4] = vld1q_f32((SRC0) + 2 * (LDS)); \
    vi[5] = vld1q_f32((SRC1) + 2 * (LDS)); \
    vi[6] = vld1q_f32((SRC2) + 2 * (LDS)); \
    vi[7] = vld1q_f32((SRC3) + 2 * (LDS)); \
    CONV_TRANSPOSE_4X4(vi[4], vi[5], vi[6], vi[7]); \
    vr[0] = vsubq_f32(vi[0], vi[4]); \
    vr[1] = vsubq_f32(vi[1], vi[5]); \
    vr[2] = vsubq_f32(vi[2], vi[6]); \
    vr[3] = vsubq_f32(vi[3], vi[7]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 0 * (LDD), vr[4]); \
    vst1q_f32((DST) + 1 * (LDD), vr[5]); \
    vst1q_f32((DST) + 2 * (LDD), vr[6]); \
    vst1q_f32((DST) + 3 * (LDD), vr[7]); \
    vi[0] = vld1q_f32((SRC0) + 1 * (LDS)); \
    vi[1] = vld1q_f32((SRC1) + 1 * (LDS)); \
    vi[2] = vld1q_f32((SRC2) + 1 * (LDS)); \
    vi[3] = vld1q_f32((SRC3) + 1 * (LDS)); \
    CONV_TRANSPOSE_4X4(vi[0], vi[1], vi[2], vi[3]); \
    vr[0] = vaddq_f32(vi[0], vi[4]); \
    vr[1] = vaddq_f32(vi[1], vi[5]); \
    vr[2] = vaddq_f32(vi[2], vi[6]); \
    vr[3] = vaddq_f32(vi[3], vi[7]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 4 * (LDD), vr[4]); \
    vst1q_f32((DST) + 5 * (LDD), vr[5]); \
    vst1q_f32((DST) + 6 * (LDD), vr[6]); \
    vst1q_f32((DST) + 7 * (LDD), vr[7]); \
    vr[0] = vsubq_f32(vi[4], vi[0]); \
    vr[1] = vsubq_f32(vi[5], vi[1]); \
    vr[2] = vsubq_f32(vi[6], vi[2]); \
    vr[3] = vsubq_f32(vi[7], vi[3]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 8 * (LDD), vr[4]); \
    vst1q_f32((DST) + 9 * (LDD), vr[5]); \
    vst1q_f32((DST) + 10 * (LDD), vr[6]); \
    vst1q_f32((DST) + 11 * (LDD), vr[7]); \
    vi[4] = vld1q_f32((SRC0) + 3 * (LDS)); \
    vi[5] = vld1q_f32((SRC1) + 3 * (LDS)); \
    vi[6] = vld1q_f32((SRC2) + 3 * (LDS)); \
    vi[7] = vld1q_f32((SRC3) + 3 * (LDS)); \
    CONV_TRANSPOSE_4X4(vi[4], vi[5], vi[6], vi[7]); \
    vr[0] = vsubq_f32(vi[0], vi[4]); \
    vr[1] = vsubq_f32(vi[1], vi[5]); \
    vr[2] = vsubq_f32(vi[2], vi[6]); \
    vr[3] = vsubq_f32(vi[3], vi[7]); \
    vr[4] = vsubq_f32(vr[0], vr[2]); \
    vr[5] = vaddq_f32(vr[1], vr[2]); \
    vr[6] = vsubq_f32(vr[2], vr[1]); \
    vr[7] = vsubq_f32(vr[1], vr[3]); \
    vst1q_f32((DST) + 12 * (LDD), vr[4]); \
    vst1q_f32((DST) + 13 * (LDD), vr[5]); \
    vst1q_f32((DST) + 14 * (LDD), vr[6]); \
    vst1q_f32((DST) + 15 * (LDD), vr[7]); \
}

static void conv_wgb2_f3s1_src_trans_col_w12(float *sp[],
    int src_pad_h,
    int src_pad_w,
    int channels,
    int trans_align,
    float *src_trans)
{
    int i, j;

    int k = WINO_PADDING(channels, 4);

    for (i = 0; i < channels; i++)
    {
        CONV_WGB2_F3S1_SRC_TRANS(sp[0], sp[1], sp[2],
            sp[3], src_pad_w, src_trans + 0, trans_align);
        CONV_WGB2_F3S1_SRC_TRANS(sp[4], sp[5], sp[6],
            sp[7], src_pad_w, src_trans + 4, trans_align);
        CONV_WGB2_F3S1_SRC_TRANS(sp[8], sp[9], sp[10],
            sp[11], src_pad_w, src_trans + 8, trans_align);

        for (j = 0; j < CONV_WG_TRANS_SIZE; j++)
        {
            sp[j] += src_pad_h * src_pad_w;
        }
        src_trans += CONV_WG_TRANS_SIZE;
    }
    if (i < k)
    {
        for (j = 0; j < 16; j++)
        {
            memset(src_trans + j * trans_align, 0,
                (k - i) * CONV_WG_TRANS_SIZE * sizeof(float));
        }
    }
}

static void conv_wgb2_f3s1_src_trans_col_w8(float *sp[],
    int src_pad_h,
    int src_pad_w,
    int channels,
    int trans_align,
    float *src_trans)
{
    int i, j;

    int k = WINO_PADDING(channels, 4);

    for (i = 0; i < channels; i++)
    {
        CONV_WGB2_F3S1_SRC_TRANS(sp[0], sp[1], sp[2],
            sp[3], src_pad_w, src_trans + 0, trans_align);
        CONV_WGB2_F3S1_SRC_TRANS(sp[4], sp[5], sp[6],
            sp[7], src_pad_w, src_trans + 4, trans_align);

        for (j = 0; j < 8; j++)
        {
            sp[j] += src_pad_h * src_pad_w;
        }
        src_trans += 8;
    }
    if (i < k)
    {
        for (j = 0; j < 16; j++)
        {
            memset(src_trans + j * trans_align,
                0, (k - i) * 8 * sizeof(float));
        }
    }
}

static void conv_wgb2_f3s1_src_trans_col_w4(float *sp[],
    int src_pad_h,
    int src_pad_w,
    int channels,
    int trans_align,
    float *src_trans)
{
    int i, j;

    int k = WINO_PADDING(channels, 4);

    for (i = 0; i < channels; i++)
    {
        CONV_WGB2_F3S1_SRC_TRANS(sp[0], sp[1], sp[2],
            sp[3], src_pad_w, src_trans + 0, trans_align);

        for (j = 0; j < 4; j++)
        {
            sp[j] += src_pad_h * src_pad_w;
        }
        src_trans += 4;
    }
    if (i < k)
    {
        for (j = 0; j < 16; j++)
        {
            memset(src_trans + j * trans_align,
                0, (k - i) * 4 * sizeof(float));
        }
    }
}

static void conv_wgb2_f3s1_src_trans(float *src_pad,
    int src_pad_h,
    int src_pad_w,
    int channels,
    float *src_trans)
{
    int i, j;

    float *sp[CONV_WG_REG_N_SIZE];

    int k = WINO_PADDING(channels, 4);
    int src_trans_area = (src_pad_h - 3 + 1) *
        (src_pad_w - 3 + 1) / 4;
    int trans_align = k * WINO_PADDING(src_trans_area, 4);

    int blk_cnt = 0;
    for (i = 0; i <= src_pad_h - 4; i += 2)
    {
        for (j = 0; j <= src_pad_w - 4; j += 2)
        {
            sp[blk_cnt++] = src_pad + i * src_pad_w + j;

            if (blk_cnt == CONV_WG_TRANS_SIZE)
            {
                conv_wgb2_f3s1_src_trans_col_w12(sp,
                    src_pad_h, src_pad_w, channels,
                    trans_align, src_trans);

                src_trans += k * CONV_WG_TRANS_SIZE;
                blk_cnt = 0;
            }
        }
    }
    if (blk_cnt > 8)
    {
        for (i = blk_cnt; i < CONV_WG_TRANS_SIZE; i++)
        {
            sp[i] = sp[i - 1];
        }
        conv_wgb2_f3s1_src_trans_col_w12(sp, src_pad_h,
            src_pad_w, channels, trans_align, src_trans);
    }
    else if (blk_cnt > 4)
    {
        for (i = blk_cnt; i < 8; i++)
        {
            sp[i] = sp[i - 1];
        }
        conv_wgb2_f3s1_src_trans_col_w8(sp, src_pad_h,
            src_pad_w, channels, trans_align, src_trans);
    }
    else if (blk_cnt > 0)
    {
        for (i = blk_cnt; i < 4; i++)
        {
            sp[i] = sp[i - 1];
        }
        conv_wgb2_f3s1_src_trans_col_w4(sp, src_pad_h,
            src_pad_w, channels, trans_align, src_trans);
    }
}

#define CONV_WGB2_F3S1_DST_TRANS(SRC, LDS, DST, LDD, LINE) \
{ \
    float32x4x2_t vo[2]; \
    float32x4_t vr[12]; \
    vr[0] = vld1q_f32((SRC) + 0 * (LDS)); \
    vr[1] = vld1q_f32((SRC) + 1 * (LDS)); \
    vr[2] = vld1q_f32((SRC) + 2 * (LDS)); \
    vr[3] = vld1q_f32((SRC) + 3 * (LDS)); \
    vr[4] = vld1q_f32((SRC) + 4 * (LDS)); \
    vr[5] = vld1q_f32((SRC) + 5 * (LDS)); \
    vr[6] = vld1q_f32((SRC) + 6 * (LDS)); \
    vr[7] = vld1q_f32((SRC) + 7 * (LDS)); \
    vr[8] = vaddq_f32(vr[0], vr[4]); \
    vr[9] = vaddq_f32(vr[1], vr[5]); \
    vr[10] = vaddq_f32(vr[2], vr[6]); \
    vr[11] = vaddq_f32(vr[3], vr[7]); \
    vr[0] = vld1q_f32((SRC) + 8 * (LDS)); \
    vr[1] = vld1q_f32((SRC) + 9 * (LDS)); \
    vr[2] = vld1q_f32((SRC) + 10 * (LDS)); \
    vr[3] = vld1q_f32((SRC) + 11 * (LDS)); \
    vr[8] = vaddq_f32(vr[0], vr[8]); \
    vr[9] = vaddq_f32(vr[1], vr[9]); \
    vr[10] = vaddq_f32(vr[2], vr[10]); \
    vr[11] = vaddq_f32(vr[3], vr[11]); \
    vo[0].val[0] = vaddq_f32(vr[8], vr[9]); \
    vo[0].val[1] = vsubq_f32(vr[9], vr[10]); \
    vo[0].val[0] = vaddq_f32(vo[0].val[0], vr[10]); \
    vo[0].val[1] = vsubq_f32(vo[0].val[1], vr[11]); \
    vr[0] = vsubq_f32(vr[4], vr[0]); \
    vr[1] = vsubq_f32(vr[5], vr[1]); \
    vr[2] = vsubq_f32(vr[6], vr[2]); \
    vr[3] = vsubq_f32(vr[7], vr[3]); \
    vr[4] = vld1q_f32((SRC) + 12 * (LDS)); \
    vr[5] = vld1q_f32((SRC) + 13 * (LDS)); \
    vr[6] = vld1q_f32((SRC) + 14 * (LDS)); \
    vr[7] = vld1q_f32((SRC) + 15 * (LDS)); \
    vr[0] = vsubq_f32(vr[0], vr[4]); \
    vr[1] = vsubq_f32(vr[1], vr[5]); \
    vr[2] = vsubq_f32(vr[2], vr[6]); \
    vr[3] = vsubq_f32(vr[3], vr[7]); \
    vo[1].val[0] = vaddq_f32(vr[0], vr[1]); \
    vo[1].val[1] = vsubq_f32(vr[1], vr[2]); \
    vo[1].val[0] = vaddq_f32(vo[1].val[0], vr[2]); \
    vo[1].val[1] = vsubq_f32(vo[1].val[1], vr[3]); \
    vo[0].val[0] = vaddq_f32(vo[0].val[0], vbias); \
    vo[0].val[1] = vaddq_f32(vo[0].val[1], vbias); \
    vo[1].val[0] = vaddq_f32(vo[1].val[0], vbias); \
    vo[1].val[1] = vaddq_f32(vo[1].val[1], vbias); \
    vst2q_f32((DST) + 0 * (LDD), vo[0]); \
    if ((LINE) > 1) vst2q_f32((DST) + 1 * (LDD), vo[1]); \
}

static void conv_wgb2_f3s1_dst_trans_row_wmax(float *dst_trans,
    int num_outs,
    int trans_align,
    float *bias,
    int dst_real_h,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    float32x4_t vbias;

    int m = WINO_PADDING(num_outs, 4);

    for (i = 0; i < num_outs; i++)
    {
        float *dst_trans_d = dst_trans;
        float *dst_d = dst;

        vbias = vdupq_n_f32(bias[i]);
        for (j = 0; j <= dst_real_h - 2; j += 2)
        {
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 0,
                trans_align, dst_d +  0, dst_w, 2);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 4,
                trans_align, dst_d +  8, dst_w, 2);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 8,
                trans_align, dst_d + 16, dst_w, 2);

            dst_trans_d += m * CONV_WG_TRANS_SIZE;
            dst_d += dst_w * 2;
        }
        if (j < dst_real_h)
        {
            // dst_real_h - j = 1
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 0,
                trans_align, dst_d +  0, dst_w, 1);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 4,
                trans_align, dst_d +  8, dst_w, 1);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_d + 8,
                trans_align, dst_d + 16, dst_w, 1);
        }

        dst_trans += CONV_WG_TRANS_SIZE;
        dst += dst_h * dst_w;
    }
}

static inline void conv_wgb2_f3s1_dst_trans_hwmax(float *dst_trans,
    int num_outs,
    float *bias,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i;

    int trans_align = CONV_WGB2_F3S1_N_SIZE *
        WINO_PADDING(num_outs, 4);

    for (i = 0; i <= num_outs - CONV_WG_REG_M_SIZE;
        i += CONV_WG_REG_M_SIZE)
    {
        conv_wgb2_f3s1_dst_trans_row_wmax(dst_trans,
            CONV_WG_REG_M_SIZE, trans_align, bias + i,
            CONV_WG_DST_PAD_SIZE, dst_h, dst_w, dst);

        dst_trans += CONV_WG_REG_M_SIZE *
            CONV_WGB2_F3S1_N_SIZE;
        dst += CONV_WG_REG_M_SIZE * dst_h * dst_w;
    }
    if (i < num_outs)
    {
        conv_wgb2_f3s1_dst_trans_row_wmax(dst_trans,
            num_outs - i, trans_align, bias + i,
            CONV_WG_DST_PAD_SIZE, dst_h, dst_w, dst);
    }
}

static inline void conv_wgb2_f3s1_dst_trans_wmax(float *dst_trans,
    int num_outs,
    float *bias,
    int dst_real_h,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i;

    int n = CONV_WG_TRANS_SIZE *
        WINO_PADDING(dst_real_h, 2) / 2;
    int trans_align = n * WINO_PADDING(num_outs, 4);

    for (i = 0; i <= num_outs - CONV_WG_REG_M_SIZE;
        i += CONV_WG_REG_M_SIZE)
    {
        conv_wgb2_f3s1_dst_trans_row_wmax(dst_trans,
            CONV_WG_REG_M_SIZE, trans_align, bias + i,
            dst_real_h, dst_h, dst_w, dst);

        dst_trans += n * CONV_WG_REG_M_SIZE;
        dst += CONV_WG_REG_M_SIZE * dst_h * dst_w;
    }
    if (i < num_outs)
    {
        conv_wgb2_f3s1_dst_trans_row_wmax(dst_trans,
            num_outs - i, trans_align, bias + i,
            dst_real_h, dst_h, dst_w, dst);
    }
}

#define CONV_WGB2_F3S1_DST_TRANS_TODST(DB, LDB, DST, LDD) \
{ \
    float32x4_t vr[2]; \
    for (j = 0; j <= dst_real_h - 2; j += 2) \
    { \
        for (ii = 0; ii <= dst_real_w - 4; ii += 4) \
        { \
            vr[0] = vld1q_f32((DB) + 0 * (LDB) + ii); \
            vr[1] = vld1q_f32((DB) + 1 * (LDB) + ii); \
            vst1q_f32((DST) + 0 * (LDD) + ii, vr[0]); \
            vst1q_f32((DST) + 1 * (LDD) + ii, vr[1]); \
        } \
        for (; ii < dst_real_w; ii++) \
        { \
            (DST)[0 * (LDD) + ii] = (DB)[ii]; \
            (DST)[1 * (LDD) + ii] = (DB)[(LDB) + ii]; \
        } \
        (DB) += dst_pad_w; \
        (DST) += 2 * (LDD); \
    } \
    if (j < dst_real_h) \
    { \
        for (ii = 0; ii <= dst_real_w - 4; ii += 4) \
        { \
            vr[0] = vld1q_f32((DB) + ii); \
            vst1q_f32((DST) + ii, vr[0]); \
        } \
        for (; ii < dst_real_w; ii++) \
        { \
            (DST)[ii] = (DB)[ii]; \
        } \
    } \
}

static void conv_wgb2_f3s1_dst_trans_row(float *dst_trans,
    int num_outs,
    int trans_align,
    float *buffer,
    float *bias,
    int dst_real_h,
    int dst_real_w,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j, ii;

    float32x4_t vbias;

    int dst_pad_h = WINO_PADDING(dst_real_h, 2);
    int dst_pad_w = WINO_PADDING(dst_real_w, 2);

    int n = WINO_PADDING(dst_pad_h * dst_pad_w / 4, 4);

    int buf_align = n * 2;
    int reg_align = CONV_WG_TRANS_SIZE *
        WINO_PADDING(num_outs, 4);

    for (i = 0; i < num_outs; i++)
    {
        float *dst_trans_d = dst_trans;
        float *dbuf = buffer;

        vbias = vdupq_n_f32(bias[i]);
        for (j = 0; j <= n - CONV_WG_TRANS_SIZE;
            j += CONV_WG_TRANS_SIZE)
        {
            float *dst_trans_dd = dst_trans_d +
                i * CONV_WG_TRANS_SIZE;

            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 0,
                trans_align, dbuf +  0, buf_align, 2);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 4,
                trans_align, dbuf +  8, buf_align, 2);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 8,
                trans_align, dbuf + 16, buf_align, 2);

            dst_trans_d += reg_align;
            dbuf += CONV_WG_DST_PAD_SIZE;
        }
        if (n - j == 8)
        {
            float *dst_trans_dd = dst_trans_d + i * 8;

            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 0,
                trans_align, dbuf +  0, buf_align, 2);
            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 4,
                trans_align, dbuf +  8, buf_align, 2);
        }
        else if (n - j == 4)
        {
            float *dst_trans_dd = dst_trans_d + i * 4;

            CONV_WGB2_F3S1_DST_TRANS(dst_trans_dd + 0,
                trans_align, dbuf +  0, buf_align, 2);
        }

        float *dst_d = dst;
        dbuf = buffer;
        CONV_WGB2_F3S1_DST_TRANS_TODST(dbuf, buf_align,
            dst_d, dst_w);

        dst += dst_h * dst_w;
    }
}

static inline void conv_wgb2_f3s1_dst_trans(float *dst_trans,
    int num_outs,
    float *bias,
    float *buffer,
    int dst_real_h,
    int dst_real_w,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i;

    int dst_pad_h = WINO_PADDING(dst_real_h, 2);
    int dst_pad_w = WINO_PADDING(dst_real_w, 2);

    int n = WINO_PADDING(dst_pad_h * dst_pad_w / 4, 4);
    int trans_align = n * WINO_PADDING(num_outs, 4);

    for (i = 0; i <= num_outs - CONV_WG_REG_M_SIZE;
        i += CONV_WG_REG_M_SIZE)
    {
        conv_wgb2_f3s1_dst_trans_row(dst_trans,
            CONV_WG_REG_M_SIZE, trans_align, buffer,
            bias + i, dst_real_h, dst_real_w, dst_h,
            dst_w, dst);

        dst_trans += CONV_WG_REG_M_SIZE * n;
        dst += CONV_WG_REG_M_SIZE * dst_h * dst_w;
    }
    if (i < num_outs)
    {
        conv_wgb2_f3s1_dst_trans_row(dst_trans,
            num_outs - i, trans_align, buffer, bias + i,
            dst_real_h, dst_real_w, dst_h, dst_w, dst);
    }
}

static inline void conv_wgb2_batch_sgemm_kernel_m1(float *filters,
    int flt_align,
    float *src_trans,
    int num_outs,
    int num_blks,
    int channels,
    float *dst_trans)
{
    int i;

    int m = WINO_PADDING(num_outs, 4);
    int n = WINO_PADDING(num_blks, 4);
    int k = WINO_PADDING(channels, 4);

    for (i = 0; i < 16; i++)
    {
        sgemm_kernel_fp32_m1(m, n, k, filters,
            src_trans, dst_trans);

        filters += flt_align;
        src_trans += k * n;
        dst_trans += m * n;
    }
}

static inline bool conv_wg_blocking_image(float *src,
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
        CONV_WG_SRC_PAD_SIZE);
    real_dst_w = MIN_INT(hori_end - w_idx,
        CONV_WG_SRC_PAD_SIZE);

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

static void conv_wgb2_f3s1_block_padding_hwmax(float *src_pad,
    int channels,
    float *filters,
    float *bias,
    int num_outs,
    float *src_trans,
    float *dst_trans,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    /* transform blocked input */
    float *src_trans_d = src_trans;
    for (i = 0; i <= channels - CONV_WGB2_F3S1_K_SIZE;
        i += CONV_WGB2_F3S1_K_SIZE)
    {
        conv_wgb2_f3s1_src_trans_hwmax(src_pad,
            CONV_WGB2_F3S1_K_SIZE, src_trans_d);

        src_pad += CONV_WGB2_F3S1_K_SIZE *
            CONV_WG_SRC_PAD_SIZE *
            CONV_WG_SRC_PAD_SIZE;
        src_trans_d += 16 * CONV_WGB2_F3S1_K_SIZE *
            CONV_WGB2_F3S1_N_SIZE;
    }
    if (i < channels)
    {
        conv_wgb2_f3s1_src_trans_hwmax(src_pad,
            channels - i, src_trans_d);
    }

    /* dot-product */
    for (i = 0; i <= num_outs - CONV_WGB2_F3S1_M_SIZE;
        i += CONV_WGB2_F3S1_M_SIZE)
    {
        memset(dst_trans, 0, 16 * CONV_WGB2_F3S1_M_SIZE *
            CONV_WGB2_F3S1_N_SIZE * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = CONV_WGB2_F3S1_M_SIZE *
                CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                CONV_WGB2_F3S1_N_SIZE, CONV_WGB2_F3S1_K_SIZE,
                dst_trans);

            src_trans_d += 16 * CONV_WGB2_F3S1_K_SIZE *
                CONV_WGB2_F3S1_N_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = k * CONV_WGB2_F3S1_M_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                CONV_WGB2_F3S1_N_SIZE, k, dst_trans);

            filters += 16 * flt_align;
        }

        conv_wgb2_f3s1_dst_trans_hwmax(dst_trans,
            CONV_WGB2_F3S1_M_SIZE, bias + i, dst_h,
            dst_w, dst + i * dst_h * dst_w);
    }
    if (i < num_outs)
    {
        int m = WINO_PADDING(num_outs - i, 4);

        memset(dst_trans, 0, 16 * m *
            CONV_WGB2_F3S1_N_SIZE * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = m * CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, CONV_WGB2_F3S1_N_SIZE,
                CONV_WGB2_F3S1_K_SIZE, dst_trans);

            src_trans_d += 16 * CONV_WGB2_F3S1_K_SIZE *
                CONV_WGB2_F3S1_N_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = m * k;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, CONV_WGB2_F3S1_N_SIZE,
                k, dst_trans);
        }

        conv_wgb2_f3s1_dst_trans_hwmax(dst_trans,
            num_outs - i, bias + i, dst_h, dst_w,
            dst + i * dst_h * dst_w);
    }
}

static void conv_wgb2_f3s1_block_padding_wmax(float *src_pad,
    int channels,
    float *filters,
    float *bias,
    int num_outs,
    float *src_trans,
    float *dst_trans,
    int dst_real_h,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    int dst_pad_h = WINO_PADDING(dst_real_h, 2);
    int src_pad_h = dst_pad_h + 3 - 1;
    int n = dst_pad_h * CONV_WG_TRANS_SIZE / 2;

    /* transform blocked input */
    float *src_trans_d = src_trans;
    for (i = 0; i <= channels - CONV_WGB2_F3S1_K_SIZE;
        i += CONV_WGB2_F3S1_K_SIZE)
    {
        conv_wgb2_f3s1_src_trans_wmax(src_pad, src_pad_h,
            CONV_WGB2_F3S1_K_SIZE, src_trans_d);

        src_pad += src_pad_h * CONV_WGB2_F3S1_K_SIZE *
            CONV_WG_SRC_PAD_SIZE;
        src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
    }
    if (i < channels)
    {
        conv_wgb2_f3s1_src_trans_wmax(src_pad, src_pad_h,
            channels - i, src_trans_d);
    }

    /* dot-product */
    for (i = 0; i <= num_outs - CONV_WGB2_F3S1_M_SIZE;
        i += CONV_WGB2_F3S1_M_SIZE)
    {
        memset(dst_trans, 0, 16 * n *
            CONV_WGB2_F3S1_M_SIZE * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = CONV_WGB2_F3S1_M_SIZE *
                CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                n, CONV_WGB2_F3S1_K_SIZE, dst_trans);

            src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = k * CONV_WGB2_F3S1_M_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                n, k, dst_trans);

            filters += 16 * flt_align;
        }

        conv_wgb2_f3s1_dst_trans_wmax(dst_trans,
            CONV_WGB2_F3S1_M_SIZE, bias + i, dst_real_h,
            dst_h, dst_w, dst + i * dst_h * dst_w);
    }
    if (i < num_outs)
    {
        int m = WINO_PADDING(num_outs - i, 4);

        memset(dst_trans, 0, 16 * m * n * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = m * CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, n, CONV_WGB2_F3S1_K_SIZE,
                dst_trans);

            src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = m * k;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, n, k, dst_trans);
        }

        conv_wgb2_f3s1_dst_trans_wmax(dst_trans,
            num_outs - i, bias + i, dst_real_h,
            dst_h, dst_w, dst + i * dst_h * dst_w);
    }
}

static void conv_wgb2_f3s1_block_padding(float *src_pad,
    int channels,
    float *filters,
    float *bias,
    int num_outs,
    float *src_trans,
    float *dst_trans,
    float *buffer,
    int dst_real_h,
    int dst_real_w,
    int dst_h,
    int dst_w,
    float *dst)
{
    int i, j;

    int dst_pad_h = WINO_PADDING(dst_real_h, 2);
    int dst_pad_w = WINO_PADDING(dst_real_w, 2);

    int src_pad_h = dst_pad_h + 3 - 1;
    int src_pad_w = dst_pad_w + 3 - 1;

    int n = WINO_PADDING(dst_pad_h * dst_pad_w / 4, 4);

    /* transform blocked input */
    float *src_trans_d = src_trans;
    for (i = 0; i <= channels - CONV_WGB2_F3S1_K_SIZE;
        i += CONV_WGB2_F3S1_K_SIZE)
    {
        conv_wgb2_f3s1_src_trans(src_pad, src_pad_h,
            src_pad_w, CONV_WGB2_F3S1_K_SIZE, src_trans_d);

        src_pad += src_pad_h * src_pad_w *
            CONV_WGB2_F3S1_K_SIZE;
        src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
    }
    if (i < channels)
    {
        conv_wgb2_f3s1_src_trans(src_pad, src_pad_h,
            src_pad_w, channels - i, src_trans_d);
    }

    /* dot-product */
    for (i = 0; i <= num_outs - CONV_WGB2_F3S1_M_SIZE;
        i += CONV_WGB2_F3S1_M_SIZE)
    {
        memset(dst_trans, 0, 16 * n *
            CONV_WGB2_F3S1_M_SIZE * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = CONV_WGB2_F3S1_M_SIZE *
                CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                n, CONV_WGB2_F3S1_K_SIZE, dst_trans);

            src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = k * CONV_WGB2_F3S1_M_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, CONV_WGB2_F3S1_M_SIZE,
                n, k, dst_trans);

            filters += 16 * flt_align;
        }

        conv_wgb2_f3s1_dst_trans(dst_trans,
            CONV_WGB2_F3S1_M_SIZE, bias + i, buffer,
            dst_real_h, dst_real_w, dst_h, dst_w,
            dst + i * dst_h * dst_w);
    }
    if (i < num_outs)
    {
        int m = WINO_PADDING(num_outs - i, 4);

        memset(dst_trans, 0, 16 * m * n * sizeof(float));

        src_trans_d = src_trans;
        for (j = 0; j <= channels - CONV_WGB2_F3S1_K_SIZE;
            j += CONV_WGB2_F3S1_K_SIZE)
        {
            int flt_align = m * CONV_WGB2_F3S1_K_SIZE;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, n, CONV_WGB2_F3S1_K_SIZE,
                dst_trans);

            src_trans_d += 16 * n * CONV_WGB2_F3S1_K_SIZE;
            filters += 16 * flt_align;
        }
        if (j < channels)
        {
            int k = WINO_PADDING(channels - j, 4);
            int flt_align = m * k;

            conv_wgb2_batch_sgemm_kernel_m1(filters, flt_align,
                src_trans_d, m, n, k, dst_trans);
        }

        conv_wgb2_f3s1_dst_trans(dst_trans, num_outs - i,
            bias + i, buffer, dst_real_h, dst_real_w,
            dst_h, dst_w, dst + i * dst_h * dst_w);
    }
}

void conv_winograd_b2f3s1_m1(float *src,
    int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    float *cvt_filter,
    float *bias,
    int num_outs,
    void *buffer,
    float *dst)
{
    int src_pad_h = src_h + 2 * padding_h;
    int src_pad_w = src_w + 2 * padding_w;

    int dst_h = src_pad_h - 3 + 1;
    int dst_w = src_pad_w - 3 + 1;

    int dst_pad_h = WINO_PADDING(dst_h, 2);
    int dst_pad_w = WINO_PADDING(dst_w, 2);

    src_pad_h = dst_pad_h + 3 - 1;
    src_pad_w = dst_pad_w + 3 - 1;

    int k = WINO_PADDING(channels, 4);

    int buffer_len = CONV_WGB2_F3S1_N_SIZE * 4;

    int src_trans_len;
    if(dst_w == CONV_WG_DST_PAD_SIZE &&
        dst_h < CONV_WG_DST_PAD_SIZE)
    {
        int n = dst_pad_h * CONV_WG_TRANS_SIZE / 2;
        src_trans_len = 16 * n * k;
    }
    else if(dst_w < CONV_WG_DST_PAD_SIZE &&
        dst_h < CONV_WG_DST_PAD_SIZE)
    {
        int n = WINO_PADDING(dst_pad_h * dst_pad_w / 4, 4);
        src_trans_len = 16 * n * k;
    }
    else
    {
        src_trans_len = 16 * CONV_WGB2_F3S1_N_SIZE * k;
    }

    float *src_trans = (float*)buffer + buffer_len;
    float *dst_trans = src_trans + src_trans_len;
    float *src_pad = dst_trans;

    if (src_pad_h <= CONV_WG_SRC_PAD_SIZE &&
        src_pad_w <= CONV_WG_SRC_PAD_SIZE)
    {
        int left = padding_w;
        int right = src_pad_w - src_w - left;
        int top = padding_h;
        int bottom = src_pad_h - src_h - top;

        conv_wg_padding_image(src, src_h, src_w,
            channels, left, right, top, bottom, src_pad);

        if (dst_h == CONV_WG_DST_PAD_SIZE &&
            dst_w == CONV_WG_DST_PAD_SIZE)
        {
            conv_wgb2_f3s1_block_padding_hwmax(src_pad,
                channels, cvt_filter, bias, num_outs,
                src_trans, dst_trans, dst_h, dst_w, dst);
        }
        else if (dst_w == CONV_WG_DST_PAD_SIZE &&
            dst_h > 0)
        {
            conv_wgb2_f3s1_block_padding_wmax(src_pad,
                channels, cvt_filter, bias, num_outs, src_trans,
                dst_trans, dst_h, dst_h, dst_w, dst);
        }
        else if (dst_w > 0 && dst_h > 0)
        {
            conv_wgb2_f3s1_block_padding(src_pad, channels,
                cvt_filter, bias, num_outs, src_trans,
                dst_trans, (float*)buffer, dst_h, dst_w,
                dst_h, dst_w, dst);
        }
    }
    else
    {
        int hori_beg = -padding_w;
        int hori_end = hori_beg + src_pad_w;

        int vert_beg = -padding_h;
        int vert_end = vert_beg + src_pad_h;

        int h_idx = vert_beg;
        int w_idx = hori_beg;

        int src_blk_h = 0, src_blk_w = 0;
        int h_dst_idx = 0, w_dst_idx = 0;

        while (conv_wg_blocking_image(src, src_h, src_w,
                channels, hori_beg, hori_end, vert_beg,
                vert_end, h_idx, w_idx, src_pad, src_blk_h,
                src_blk_w))
        {
            int dst_blk_h = src_blk_h - 3 + 1;
            int dst_blk_w = src_blk_w - 3 + 1;

            int dst_real_h = MIN_INT(dst_blk_h,
                dst_h - h_dst_idx);
            int dst_real_w = MIN_INT(dst_blk_w,
                dst_w - w_dst_idx);

            if (dst_real_h == CONV_WG_DST_PAD_SIZE &&
                dst_real_w == CONV_WG_DST_PAD_SIZE)
            {
                conv_wgb2_f3s1_block_padding_hwmax(src_pad,
                    channels, cvt_filter, bias, num_outs,
                    src_trans, dst_trans, dst_h, dst_w,
                    dst + h_dst_idx * dst_w + w_dst_idx);
            }
            else if (dst_real_w == CONV_WG_DST_PAD_SIZE &&
                dst_real_h > 0)
            {
                conv_wgb2_f3s1_block_padding_wmax(src_pad,
                    channels, cvt_filter, bias, num_outs,
                    src_trans, dst_trans, dst_real_h, dst_h,
                    dst_w, dst + h_dst_idx * dst_w + w_dst_idx);
            }
            else if (dst_real_w > 0 && dst_real_h > 0)
            {
                conv_wgb2_f3s1_block_padding(src_pad,
                    channels, cvt_filter, bias, num_outs,
                    src_trans, dst_trans, (float*)buffer,
                    dst_real_h, dst_real_w, dst_h, dst_w,
                    dst + h_dst_idx * dst_w + w_dst_idx);
            }

            w_dst_idx += dst_real_w;
            if (w_dst_idx >= dst_w)
            {
                w_dst_idx = 0;
                h_dst_idx += dst_real_h;
            }
        }
    }
}

