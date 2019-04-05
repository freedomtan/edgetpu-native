cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@libedgetpu//:header",  # buildcleaner: keep
        "@libedgetpu//:lib",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "two_models_one_tpu",
    srcs = ["two_models_one_tpu.cc"],
    deps = [
        ":utils",
        "@libedgetpu//:header",  # buildcleaner: keep
        "@libedgetpu//:lib",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "two_models_two_tpus_threaded",
    srcs = ["two_models_two_tpus_threaded.cc"],
    deps = [
        ":utils",
        "@libedgetpu//:header",  # buildcleaner: keep
        "@libedgetpu//:lib",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
