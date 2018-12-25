include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -gencode arch=compute_60,code=compute_60 -gencode arch=compute_53,code=compute_53 -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
