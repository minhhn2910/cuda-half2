In an effort to overhaul this repo, I add the original benchmarks with modification for CUDA 9x and 10x. 
I planned complete the tool and support Clang & LLVM 7.0 + CUDA 10x. It depends on my time budget in the near future. 
The main thing about cuda 9x and 10x is that they support the operator_overload in $CUDA_LOCATION/include/cuda_fp16.hpp, so most functionalities are not needed in the original 
half2_operator_overload.hpp and half_operator_overload.hpp, need to comment out the dupplicate entries for later cuda. 

Disclaimer : I don't hold any copyright for the code in the benchmarks, if any request from the authors to remove them, they will be removed from this opensource repo.

This folder contains two subsets of benchmarks:

rodinia_3.1/cuda

NVIDA_example/half2_convert
