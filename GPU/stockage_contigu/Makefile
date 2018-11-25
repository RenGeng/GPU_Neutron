SRCS = neutron_GPU.cu

CC = nvcc
CFLAGS = -O3 # -std=c11
LIBS = -lm

all: neutron_GPU

neutron_GPU : neutron_GPU.cu
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~ neutron_GPU

exec:
	./neutron_GPU