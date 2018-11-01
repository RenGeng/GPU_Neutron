SRCS = neutron2.cu
EXE_NAME = neutron-seq

CC = nvcc
CFLAGS = -O3 # -std=c11
LIBS = -lm

all: ${EXE_NAME}

% : %.cu
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~
