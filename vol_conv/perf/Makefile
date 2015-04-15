CUDA_PATH=/usr/local/cuda-6.5/
CUDNN_PATH=../../cudnn-6.5-linux-x64-v2/

CC=g++
CFLAGS=-I. -I$(CUDA_PATH)/include -I$(CUDNN_PATH) -O3
LIBS=-lcudart -lcudnn -lstdc++
LFLAGS=-L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH) $(LIBS)
MAIN_BINARIES = volumetric_main volumetric_grad_main
OBJ = 

compile: $(MAIN_BINARIES)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%_main: %_main.o $(OBJ)
	$(CXX) -o $@ $^ $(LFLAGS)

clean:
	rm *.o $(MAIN_BINARIES)
	
	