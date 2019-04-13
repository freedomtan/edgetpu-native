# edgetpu-native
edgetpu-native from https://coral.googlesource.com/edgetpu-native

Added modified `label_image for tflite` and related updates so that we can build it on Coral Dev Board.
```
bazel build edgetpu/cpp/examples/label_image:label_image
```
Then we can get
```
bazel-bin/edgetpu/cpp/examples/label_image/label_image
```

With that, we can use it run either original quantized models on CPUs or Edge TPU's canned models on TPUs. E.g., with
```
bazel-bin/edgetpu/cpp/examples/label_image/label_image \
-m  ~/work/edge_tpu_data/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
-i ~/work/data/grace_hopper.bmp \
-l ~/work/data/labels.txt \
-c 50
```
I got
```
average time: 2.63822 ms 
0.796078: 653 653:military uniform
0.0901961: 907 907:Windsor tie
0.0156863: 458 458:bow tie, bow-tie, bowtie
0.0117647: 466 466:bulletproof vest
0.00392157: 922 922:book jacket, dust cover, dust jacket, dust wrapper
```
With
```
bazel-bin/edgetpu/cpp/examples/label_image/label_image \
-m  ~/work/edge_tpu_data/mobilenet_v1_1.0_224_quant.tflite \
-i ~/work/data/grace_hopper.bmp \
-l ~/work/data/labels.txt \
-c 50
```
I got
```
average time: 392.017 ms 
0.364706: 907 907:Windsor tie
0.364706: 653 653:military uniform
0.0431373: 668 668:mortarboard
0.0352941: 458 458:bow tie, bow-tie, bowtie
0.027451: 543 543:drumstick
```
See [here](https://github.com/freedomtan/edge_tpu_python_scripts) for scripts using [Python API](https://coral.withgoogle.com/docs/edgetpu/api-intro/)
