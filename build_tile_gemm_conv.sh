as -o sgemm_kernel_m1.o sgemm_kernel_m1.S
clang++ -O3 -pthread -c sgemm_kernel_fuse_m1.cpp
clang++ -O3 -pthread -c conv_tile_gemm_f3s1_m1.cpp
clang++ -O3 -pthread -c test_tile_gemm_conv.cpp
clang++ -O3 -pthread -o test_tg_conv test_tile_gemm_conv.o sgemm_kernel_m1.o sgemm_kernel_fuse_m1.o conv_tile_gemm_f3s1_m1.o
