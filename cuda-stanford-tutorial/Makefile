MAIN_BINARIES = $(basename $(wildcard *_main.cu))
COMPILER_FLAGS = -DDEBUG

compile: $(MAIN_BINARIES)

clean:
	rm -f $(MAIN_BINARIES)

%_main: %_main.cu
	nvcc $(COMPILER_FLAGS) $@.cu -o $@

