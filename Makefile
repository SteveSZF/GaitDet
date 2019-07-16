CC = g++
CUCC =$(CUDA_INSTALL_DIR)/bin/nvcc -m64

TENSORRT_DIR=/home/szf/TensorRT-5.0.2.6
CUDA_INSTALL_DIR=/usr/local/cuda-9.0
CUDA_LIBDIR=lib64

OPENCV_INC=`pkg-config --cflags opencv`
OPENCV_LIB=`pkg-config --libs opencv`

LIBPATHS=-L/usr/local/lib -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(TENSORRT_DIR)/lib" $(OPENCV_LIB)

TENSORRT_LIB=-lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime

INCPATHS=-I/usr/local/include -I"$(CUDA_INSTALL_DIR)/include"  -I"$(TENSORRT_DIR)/include" $(OPENCV_INC)

COMMON_FLAGS += -Wall -g -std=c++11  $(INCPATHS) -Wl,-rpath "$(TENSORRT_DIR)/lib"

CU_COMMON_FLAGS += -g -G -std=c++11  $(INCPATHS) -L"$(TENSORRT_DIR)/lib"

#CU_COMMON_FLAGS += -G -g -std=c++11
COMMON_LD_FLAGS += $(LIBPATHS)
COMMON_LIBS = -lcudnn -lcublas -lnvToolsExt -lcudart $(TENSORRT_LIB) -lnppicc -lnppial -lnppicom -lnppig -lnppidei -lpthread

CFILE=main.cpp Logger.cpp result_process.cpp Yolov3.cpp
CUFILE=deeplab.cu
OBJS=main.o Logger.o result_process.o Yolov3.o
CU_OBJS=deeplab.o
BIN=gait.bin
all:$(OBJS) $(CU_OBJS)
	$(CC) $(OBJS) $(CU_OBJS) $(COMMON_FLAGS) $(COMMON_LIBS) $(COMMON_LD_FLAGS) -o $(BIN)


%.o:%.cpp
	$(CC) -D TIME $(COMMON_FLAGS) $(COMMON_LIBS) $(COMMON_LD_FLAGS) -c $< -o $@
%.o:%.cu
	$(CUCC) -D TIME $(CU_COMMON_FLAGS) $(COMMON_LIBS) $(COMMON_LD_FLAGS) -c $< -o $@
clean:
	rm *.o $(BIN) 
