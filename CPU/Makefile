SRCS = neutron_CPU.c
EXE_NAME = neutron_CPU

CC = gcc
CFLAGS = -Wall -O3 # -std=c11
LIBS = -lm -fopenmp

all: ${EXE_NAME}

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~ neutron_CPU

exec:
	./neutron_CPU