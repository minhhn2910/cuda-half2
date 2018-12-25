CC = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp
FCUDA_IFLAG = -I${FROOT}/include -I${FROOT}/include/cuda_include -I${LAVAMD_float}

OBJDIR = objs
FCUDADIR = fcuda_gen
FCUDA_GEN = fcuda_gen_lavaMD

HEADERS = main.h kernel_gpu_cuda_wrapper.h
SOURCES = main.c kernel_gpu_cuda_wrapper.c ./util/num/num.c ./util/timer/timer.c $(FCUDADIR)/$(FCUDA_GEN).c
OBJECTS = $(addprefix $(OBJDIR)/, main.o kernel_gpu_cuda_wrapper.o num.o timer.o $(FCUDA_GEN).o)

EXE = lavaMD.exe

# link objects (binaries) together
all: $(OBJECTS) $(HEADERS)
	$(CC) $(OBJECTS) -o $(EXE) $(FLAGS) -lm -lmpfr $(OMP_LIB)

# compile function files into objects (binaries)
$(OBJDIR)/%.o: %.c $(HEADERS)
	$(CC) $< -c $(FCUDA_IFLAG) -o $@

$(OBJDIR)/%.o: $(FCUDADIR)/%.c $(HEADERS)
	$(CC) $< -c $(FCUDA_IFLAG) -o $@

$(OBJDIR)/%.o: util/num/%.c $(HEADERS)
	$(CC) $< -c $(FCUDA_IFLAG) -o $@

$(OBJDIR)/%.o: util/timer/%.c $(HEADERS)
	$(CC) $< -c $(FCUDA_IFLAG) -o $@

$(OBJECTS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# delete all object and executable files
clean:
	rm -rf $(EXE) $(OBJDIR)
