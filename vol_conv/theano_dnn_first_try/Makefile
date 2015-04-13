CUDA_PATH=/usr/local/cuda-6.5/
CUDNN_PATH=../../cudnn-6.5-linux-x64-v2/

CC=gcc
CFLAGS=-I. -I$(CUDA_PATH)/include -I$(CUDNN_PATH)
LIBS=-lcudart -lcudnn -lstdc++
LFLAGS=-L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH) $(LIBS)

OBJ = volumetric_in_c.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

volumetric_in_c: $(OBJ)
	gcc -o $@ $^ $(LFLAGS)

clean:
	rm *.o ./mnistCUDNN