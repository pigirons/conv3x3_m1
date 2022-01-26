as -o sgemm_kernel_aarch64.o sgemm_kernel_k4_aarch64.S
#as -o sgemm_kernel_aarch64.o sgemm_kernel_k1_aarch64.S
clang -O3 -pthread -c test_sgemm_kernel.c
clang -O3 -pthread -o test_sk test_sgemm_kernel.o sgemm_kernel_aarch64.o
