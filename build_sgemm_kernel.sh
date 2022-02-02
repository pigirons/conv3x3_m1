as -o sgemm_kernel_m1.o sgemm_kernel_m1.S
clang++ -O3 -pthread -c sgemm_kernel_fuse_m1.cpp
clang++ -O3 -pthread -c test_sgemm_kernel.cpp
clang++ -O3 -pthread -o test_sk test_sgemm_kernel.o sgemm_kernel_m1.o sgemm_kernel_fuse_m1.cpp
