as -o sgemm_kernel_m1.o sgemm_kernel_m1.S
clang++ -O3 -pthread -c sgemm_kernel_fuse_m1.cpp
clang++ -O3 -pthread -c conv_winograd_b2f3s1_m1.cpp
clang++ -O3 -pthread -c test_winograd_conv.cpp
clang++ -O3 -pthread -o test_wg_conv test_winograd_conv.o sgemm_kernel_m1.o sgemm_kernel_fuse_m1.o conv_winograd_b2f3s1_m1.o
