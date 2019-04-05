workspace(name = "edgetpu")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "43c9b882fa921923bcba764453f4058d102bece35a37c9f6383c713004aacff1",
    strip_prefix = "rules_closure-9889e2348259a5aad7e805547c1a0cf311cfcd91",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
    ],
)
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
closure_repositories()

http_archive(
    name = "com_google_glog",
    # For security purpose, can use `sha256sum` on linux to calculate.
    sha256 = "835888ec47ee8065b3098f3ec4373717d641954970f009833ed6d466c397409a",
    strip_prefix = "glog-41f4bf9cbc3e8995d628b459f6a239df43c2b84a",
    urls = [
        "https://github.com/google/glog/archive/41f4bf9cbc3e8995d628b459f6a239df43c2b84a.tar.gz",
    ],
)

# Make sure it matches the definition in glog/WORKSPACE
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)

http_archive(
  name = "com_google_absl",
  sha256 = "d10f684f170eb36f3ce752d2819a0be8cc703b429247d7d662ba5b4b48dd7f65",
  strip_prefix = "abseil-cpp-3088e76c597e068479e82508b1770a7ad0c806b6",
  urls = [
    "https://github.com/abseil/abseil-cpp/archive/3088e76c597e068479e82508b1770a7ad0c806b6.tar.gz",
  ],
)

local_repository(
    name = "org_tensorflow",
    path = "tensorflow",
)
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace()

local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
)

new_local_repository(
    name = "python_linux",
    path = "/",
    build_file = "BUILD.python"
)
