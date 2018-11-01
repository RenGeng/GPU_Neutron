SRCS = neutron-seq.c
EXE_NAME = neutron-seq

CC = gcc
CFLAGS = -Wall -O3 # -std=c11
LIBS = -lm

all: ${EXE_NAME}

% : %.c
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(LIBS)

clean:
	rm -f ${EXE_NAME} *.o *~