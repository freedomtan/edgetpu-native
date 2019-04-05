# Makefile for cross-compiling edgetpu/cpp/examples.
#
# Note that cross-compilation happens inside docker using bazel, this makefile
# here is mainly for convenience.
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := cpp_example_out
TAG_EXAMPLE := coral/cpp-example-cross-compile
UID ?= $(shell id -u)
GID ?= $(shell id -g)
COPY := install -d -m 755 -o $(UID) -g $(GID) $(OUT_DIR) && install -C -m 755 -o $(UID) -g $(GID)
BAZEL_FLAGS := -c opt \
               --verbose_failures \
               --sandbox_debug \
               --crosstool_top=//tools/arm_compiler:toolchain \
               --compiler=clang


.PHONY: all \
        docker-cpp-example-image \
        docker-cpp-example-shell \
        docker-cpp-example-compile \
        amd64 \
        arm64 \
        arm32 \
        clean

all:
	@echo "make docker-cpp-example-image   - Build docker image for cpp examples"
	@echo "make docker-cpp-example-shell   - Run shell to docker image for cpp examples"
	@echo "make docker-cpp-example-compile - Compile cpp examples for all platforms"
	@echo "make cpp-examples               - Compile cpp examples for amd64, arm64, arm32"
	@echo "make clean               - Remove generated files"

docker-cpp-example-image:
	docker build -t $(TAG_EXAMPLE) -f tools/Dockerfile.16.04 tools

docker-cpp-example-shell: docker-cpp-example-image
	docker run --rm -it -v $(ROOT_DIR):/edgetpu-ml-cpp $(TAG_EXAMPLE)

docker-cpp-example-compile: docker-cpp-example-image
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-ml-cpp $(TAG_EXAMPLE) make -f cpp-example.makefile UID=$(UID) GID=$(GID) -C /edgetpu-ml-cpp cpp-examples

cpp-examples:
	bazel build $(BAZEL_FLAGS) --features=glibc_compat --cpu=k8 //edgetpu/cpp/examples:two_models_one_tpu
	$(COPY) bazel-out/k8-opt/bin/edgetpu/cpp/examples/two_models_one_tpu $(OUT_DIR)/two_models_one_tpu_amd64
	bazel build $(BAZEL_FLAGS) --features=glibc_compat --cpu=k8 //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(COPY) bazel-out/k8-opt/bin/edgetpu/cpp/examples/two_models_two_tpus_threaded $(OUT_DIR)/two_models_two_tpus_threaded_amd64
	bazel build $(BAZEL_FLAGS) --cpu=arm64-v8a //edgetpu/cpp/examples:two_models_one_tpu
	$(COPY) bazel-out/arm64-v8a-opt/bin/edgetpu/cpp/examples/two_models_one_tpu $(OUT_DIR)/two_models_one_tpu_arm64
	bazel build $(BAZEL_FLAGS) --cpu=arm64-v8a //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(COPY) bazel-out/arm64-v8a-opt/bin/edgetpu/cpp/examples/two_models_two_tpus_threaded $(OUT_DIR)/two_models_two_tpus_threaded_arm64
	bazel build $(BAZEL_FLAGS) --cpu=armeabi-v7a //edgetpu/cpp/examples:two_models_one_tpu
	$(COPY) bazel-out/armeabi-v7a-opt/bin/edgetpu/cpp/examples/two_models_one_tpu $(OUT_DIR)/two_models_one_tpu_arm32
	bazel build $(BAZEL_FLAGS) --cpu=armeabi-v7a //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(COPY) bazel-out/armeabi-v7a-opt/bin/edgetpu/cpp/examples/two_models_two_tpus_threaded $(OUT_DIR)/two_models_two_tpus_threaded_arm32

clean:
	rm -rf $(OUT_DIR)
