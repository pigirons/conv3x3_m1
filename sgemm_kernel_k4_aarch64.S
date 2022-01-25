.align 4

.macro preserve_caller_vec
	stp d8, d9, [sp, #-16]!
	stp d10, d11, [sp, #-16]!
	stp d12, d13, [sp, #-16]!
	stp d14, d15, [sp, #-16]!
.endm

.macro restore_caller_vec
	ldp d14, d15, [sp], #16
	ldp d12, d13, [sp], #16
	ldp d10, d11, [sp], #16
	ldp d8, d9, [sp], #16
.endm

.macro SGEMM_KERNEL_M8N12K4_AARCH64_LD
    ldr q8,  [x2, #0]
    ldr q9,  [x2, #16]
    ldr q10, [x2, #32]
    ldr q11, [x2, #48]
    ldr q12, [x2, #64]
    ldr q13, [x2, #80]
    ldr q14, [x2, #96]
    ldr q15, [x2, #112]
    ldr q16, [x2, #128]
    ldr q17, [x2, #144]
    ldr q18, [x2, #160]
    ldr q19, [x2, #176]
    ldr q20, [x2, #192]
    ldr q21, [x2, #208]
    ldr q22, [x2, #224]
    ldr q23, [x2, #240]
    ldr q24, [x2, #256]
    ldr q25, [x2, #272]
    ldr q26, [x2, #288]
    ldr q27, [x2, #304]
    ldr q28, [x2, #320]
    ldr q29, [x2, #336]
    ldr q30, [x2, #352]
    ldr q31, [x2, #368]
.endm

.macro SGEMM_KERNEL_M8N12K4_AARCH64_CP
    ldr q0, [x12, #0]
    ldr q1, [x12, #16]
    ldr q2, [x12, #32]
    ldr q3, [x10, #0]
    fmla  v8.4s, v0.4s, v3.s[0]
    fmla  v9.4s, v1.4s, v3.s[0]
    fmla v10.4s, v2.4s, v3.s[0]
    fmla v11.4s, v0.4s, v3.s[1]
    fmla v12.4s, v1.4s, v3.s[1]
    fmla v13.4s, v2.4s, v3.s[1]
    fmla v14.4s, v0.4s, v3.s[2]
    fmla v15.4s, v1.4s, v3.s[2]
    fmla v16.4s, v2.4s, v3.s[2]
    fmla v17.4s, v0.4s, v3.s[3]
    fmla v18.4s, v1.4s, v3.s[3]
    fmla v19.4s, v2.4s, v3.s[3]
    ldr q3, [x10, #16]
    fmla v20.4s, v0.4s, v3.s[0]
    fmla v21.4s, v1.4s, v3.s[0]
    fmla v22.4s, v2.4s, v3.s[0]
    fmla v23.4s, v0.4s, v3.s[1]
    fmla v24.4s, v1.4s, v3.s[1]
    fmla v25.4s, v2.4s, v3.s[1]
    fmla v26.4s, v0.4s, v3.s[2]
    fmla v27.4s, v1.4s, v3.s[2]
    fmla v28.4s, v2.4s, v3.s[2]
    fmla v29.4s, v0.4s, v3.s[3]
    fmla v30.4s, v1.4s, v3.s[3]
    fmla v31.4s, v2.4s, v3.s[3]
    ldr q0, [x12, #48]
    ldr q1, [x12, #64]
    ldr q2, [x12, #80]
    ldr q3, [x10, #32]
    fmla  v8.4s, v0.4s, v3.s[0]
    fmla  v9.4s, v1.4s, v3.s[0]
    fmla v10.4s, v2.4s, v3.s[0]
    fmla v11.4s, v0.4s, v3.s[1]
    fmla v12.4s, v1.4s, v3.s[1]
    fmla v13.4s, v2.4s, v3.s[1]
    fmla v14.4s, v0.4s, v3.s[2]
    fmla v15.4s, v1.4s, v3.s[2]
    fmla v16.4s, v2.4s, v3.s[2]
    fmla v17.4s, v0.4s, v3.s[3]
    fmla v18.4s, v1.4s, v3.s[3]
    fmla v19.4s, v2.4s, v3.s[3]
    ldr q3, [x10, #48]
    fmla v20.4s, v0.4s, v3.s[0]
    fmla v21.4s, v1.4s, v3.s[0]
    fmla v22.4s, v2.4s, v3.s[0]
    fmla v23.4s, v0.4s, v3.s[1]
    fmla v24.4s, v1.4s, v3.s[1]
    fmla v25.4s, v2.4s, v3.s[1]
    fmla v26.4s, v0.4s, v3.s[2]
    fmla v27.4s, v1.4s, v3.s[2]
    fmla v28.4s, v2.4s, v3.s[2]
    fmla v29.4s, v0.4s, v3.s[3]
    fmla v30.4s, v1.4s, v3.s[3]
    fmla v31.4s, v2.4s, v3.s[3]
    ldr q0, [x12, #96]
    ldr q1, [x12, #112]
    ldr q2, [x12, #128]
    ldr q3, [x10, #64]
    fmla  v8.4s, v0.4s, v3.s[0]
    fmla  v9.4s, v1.4s, v3.s[0]
    fmla v10.4s, v2.4s, v3.s[0]
    fmla v11.4s, v0.4s, v3.s[1]
    fmla v12.4s, v1.4s, v3.s[1]
    fmla v13.4s, v2.4s, v3.s[1]
    fmla v14.4s, v0.4s, v3.s[2]
    fmla v15.4s, v1.4s, v3.s[2]
    fmla v16.4s, v2.4s, v3.s[2]
    fmla v17.4s, v0.4s, v3.s[3]
    fmla v18.4s, v1.4s, v3.s[3]
    fmla v19.4s, v2.4s, v3.s[3]
    ldr q3, [x10, #80]
    fmla v20.4s, v0.4s, v3.s[0]
    fmla v21.4s, v1.4s, v3.s[0]
    fmla v22.4s, v2.4s, v3.s[0]
    fmla v23.4s, v0.4s, v3.s[1]
    fmla v24.4s, v1.4s, v3.s[1]
    fmla v25.4s, v2.4s, v3.s[1]
    fmla v26.4s, v0.4s, v3.s[2]
    fmla v27.4s, v1.4s, v3.s[2]
    fmla v28.4s, v2.4s, v3.s[2]
    fmla v29.4s, v0.4s, v3.s[3]
    fmla v30.4s, v1.4s, v3.s[3]
    fmla v31.4s, v2.4s, v3.s[3]
    ldr q0, [x12, #144]
    ldr q1, [x12, #160]
    ldr q2, [x12, #176]
    ldr q3, [x10, #96]
    fmla  v8.4s, v0.4s, v3.s[0]
    fmla  v9.4s, v1.4s, v3.s[0]
    fmla v10.4s, v2.4s, v3.s[0]
    fmla v11.4s, v0.4s, v3.s[1]
    fmla v12.4s, v1.4s, v3.s[1]
    fmla v13.4s, v2.4s, v3.s[1]
    add x12, x12, #192
    fmla v14.4s, v0.4s, v3.s[2]
    fmla v15.4s, v1.4s, v3.s[2]
    fmla v16.4s, v2.4s, v3.s[2]
    fmla v17.4s, v0.4s, v3.s[3]
    fmla v18.4s, v1.4s, v3.s[3]
    fmla v19.4s, v2.4s, v3.s[3]
    ldr q3, [x10, #112]
    fmla v20.4s, v0.4s, v3.s[0]
    fmla v21.4s, v1.4s, v3.s[0]
    fmla v22.4s, v2.4s, v3.s[0]
    fmla v23.4s, v0.4s, v3.s[1]
    fmla v24.4s, v1.4s, v3.s[1]
    fmla v25.4s, v2.4s, v3.s[1]
    add x10, x10, #128
    fmla v26.4s, v0.4s, v3.s[2]
    fmla v27.4s, v1.4s, v3.s[2]
    fmla v28.4s, v2.4s, v3.s[2]
    fmla v29.4s, v0.4s, v3.s[3]
    fmla v30.4s, v1.4s, v3.s[3]
    fmla v31.4s, v2.4s, v3.s[3]
.endm 

.macro SGEMM_KERNEL_M8N12K4_AARCH64_ST
    str q8,  [x2, #0]
    str q9,  [x2, #16]
    str q10, [x2, #32]
    str q11, [x2, #48]
    str q12, [x2, #64]
    str q13, [x2, #80]
    str q14, [x2, #96]
    str q15, [x2, #112]
    str q16, [x2, #128]
    str q17, [x2, #144]
    str q18, [x2, #160]
    str q19, [x2, #176]
    str q20, [x2, #192]
    str q21, [x2, #208]
    str q22, [x2, #224]
    str q23, [x2, #240]
    str q24, [x2, #256]
    str q25, [x2, #272]
    str q26, [x2, #288]
    str q27, [x2, #304]
    str q28, [x2, #320]
    str q29, [x2, #336]
    str q30, [x2, #352]
    str q31, [x2, #368]
.endm 

.globl _sgemm_kernel_aarch64_fp32_m8n12
_sgemm_kernel_aarch64_fp32_m8n12:
// param x0: a_loc
// param x1: b_loc
// param x2: c_loc
// param x3: m
// param x4: n
// param x5: k
    preserve_caller_vec
.sk_m8n12k4_l1:                  // l1:  loop m
    mov x12, x1                  // x12: b_loc
    mov x11, x4                  // x11: n
.sk_m8n12k4_l2:                  // l2:  loop n
    mov x10, x0                  // x10: a_loc
    mov x13, x5                  // x13: k
    SGEMM_KERNEL_M8N12K4_AARCH64_LD
.sk_m8n12k4_l3:                  // l3:  loop k
    SGEMM_KERNEL_M8N12K4_AARCH64_CP
    subs x13, x13, #4            // x13: k -= 4
    bne .sk_m8n12k4_l3           // x13: while (k != 0)
    SGEMM_KERNEL_M8N12K4_AARCH64_ST
    add x2, x2, #384             // x2:  c_loc += 8 * 12
    subs x11, x11, #12           // x11: n -= 12
    bne .sk_m8n12k4_l2           // x11: while (n != 0)
    add x0, x0, x5, lsl#5        // x0:  a_loc += k * 8
    subs x3, x3, #8              // x3:  m -= 8
    bne .sk_m8n12k4_l1           // x3:  while (m != 0)
    restore_caller_vec
    ret

