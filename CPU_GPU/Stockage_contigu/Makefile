SRCS = neutron_CPU_GPU.cu

CC = nvcc
CFLAGS = -O3 -Xcompiler -fopenmp -lgomp # -std=c11
LIBS = -lm

all : neutron_CPU_GPU

neutron_CPU_GPU : neutron_CPU_GPU.cu
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~ ./neutron_CPU_GPU

exec:
	./neutron_CPU_GPU