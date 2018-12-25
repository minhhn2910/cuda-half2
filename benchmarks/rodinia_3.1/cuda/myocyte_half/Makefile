# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte.out:		main.o
	nvcc	main.o \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm -lcuda -lcudart \
                                -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work_2.cu \
				kernels.cu \
				file.c \
				timer.c
	nvcc	-arch=sm_60 main.cu \
				-c -O3 -g

# delete all object files
clean:
	rm *.o *.out output.txt
