# CUDA Makefile - Output binaries to /build directory

NVCC = nvcc
BUILD_DIR = build
SOURCES = $(wildcard *.cu)
TARGETS = $(patsubst %.cu,$(BUILD_DIR)/%,$(SOURCES))

# CUDA compiler flags
NVCC_FLAGS = -O2 -arch=sm_80

# Default target
all: $(BUILD_DIR) $(TARGETS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile each .cu file
$(BUILD_DIR)/%: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Individual targets for convenience
1d_tiling: $(BUILD_DIR)/1d_tiling
2d_tiling: $(BUILD_DIR)/2d_tiling
gemm_naive: $(BUILD_DIR)/gemm_naive
gemm_global_memory_coalescing: $(BUILD_DIR)/gemm_global_memory_coalescing
smem_gemm: $(BUILD_DIR)/smem_gemm
vectorize: $(BUILD_DIR)/vectorize

# Profile with Nsight Systems (system-wide trace)
# Usage: make profile-nsys FILE=gemm_naive
profile-nsys: $(BUILD_DIR)/$(FILE)
	nsys profile --stats=true -o $(BUILD_DIR)/$(FILE)_report $(BUILD_DIR)/$(FILE)

# Profile with Nsight Compute (kernel analysis)
# Usage: make profile-ncu FILE=gemm_naive
profile-ncu: $(BUILD_DIR)/$(FILE)
	ncu --set full -o $(BUILD_DIR)/$(FILE)_ncu_report $(BUILD_DIR)/$(FILE)

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild 1d_tiling 2d_tiling gemm_naive gemm_global_memory_coalescing smem_gemm vectorize profile-nsys profile-ncu
