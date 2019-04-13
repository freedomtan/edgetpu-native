# Description:
# TensorFlow Lite Example Label Image.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "label_image",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
    ],
    deps = [
        ":bitmap_helpers",
	"@libedgetpu//:header",  # buildcleaner: keep
	"@libedgetpu//:lib",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:schema_fbs_version",
        "@org_tensorflow//tensorflow/lite:string",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "@com_google_googletest//:gtest",
    ],
)
