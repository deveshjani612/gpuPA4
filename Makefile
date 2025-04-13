NVCC := nvcc
CFLAGS := -O2

all: softmax 

softmax: Softmax.cu
	$(NVCC) $(CFLAGS) Softmax.cu -o softmax

clean:
	rm -f softmax 
