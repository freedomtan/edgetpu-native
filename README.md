# edgetpu-native
edgetpu-native from https://coral.googlesource.com/edgetpu-native

See [here](https://github.com/freedomtan/edge_tpu_python_scripts) for scripts using [Python API](https://coral.withgoogle.com/docs/edgetpu/api-intro/)

Added modified `label_image for tflite` and related updates so that we can build it on Coral Dev Board.
```
bazel build edgetpu/cpp/examples/label_image:label_image
```
Then we can get
```
bazel-bin/edgetpu/cpp/examples/label_image/label_image
```

With that, we can use it to run either original quantized TFLite models on CPUs or Edge TPU's canned models on Edge TPUs. E.g., with
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
With `-v`, we can see many interesing information

```
Node   0 Operator Custom Name edgetpu-custom-op
  Inputs: 31
  Outputs: 30
I0413 08:35:26.199975    4793 request.cc:24] Adding input "input" with 150528 bytes.
I0413 08:35:26.200373    4793 request.cc:35] Adding output "MobilenetV1/Predictions/Reshape_1" with 1001 bytes.
I0413 08:35:26.201448    4793 kernel_mmu_mapper.cc:124] MmuMapper#Map() : 0000ffff8fdb3000 -> 8000000000000000 (1079 pages).
I0413 08:35:26.201536    4793 mmio_address_space.cc:47] MapMemory() page-aligned : device_address = 0x8000000000000000
I0413 08:35:26.201568    4793 driver.cc:130] Mapped params : 0xffff8fdb3000 -> 0x8000000000000000, 4415744 bytes.
I0413 08:35:26.201593    4793 driver.cc:130] Mapped params : (nil) -> 0x0000000000000000, 0 bytes.
I0413 08:35:26.201625    4793 single_tpu_request.cc:59] [0] Request constructed.
I0413 08:35:26.201671    4793 instruction_buffers.cc:32] InstructionBuffers created.
I0413 08:35:26.201694    4793 package_registry.cc:535] Created new instruction buffers.
I0413 08:35:26.201718    4793 device_buffer_mapper.cc:100] Mapped scratch : (nil) -> 0x0000000000000000, 0 bytes.
I0413 08:35:26.201745    4793 single_tpu_request.cc:324] MapDataBuffers() done.
I0413 08:35:26.201768    4793 executable_util.cc:172] Linking Parameter: 0x8000000000000000
I0413 08:35:26.201808    4793 kernel_mmu_mapper.cc:124] MmuMapper#Map() : 0000aaaae2778000 -> 8000000000800000 (2 pages).
I0413 08:35:26.201833    4793 mmio_address_space.cc:47] MapMemory() page-aligned : device_address = 0x8000000000800000
I0413 08:35:26.201860    4793 device_buffer_mapper.cc:118] Mapped instructions[0] : 0xaaaae2778000 -> 0x8000000000800000, 6832 bytes.
I0413 08:35:26.201884    4793 single_tpu_request.cc:340] MapInstructionBuffers() done.
I0413 08:35:26.201904    4793 single_tpu_request.cc:434] [0] SetState old=0, new=1.
I0413 08:35:26.201928    4793 single_tpu_request.cc:349] [0] NotifyRequestSubmitted()
I0413 08:35:26.201950    4793 single_tpu_request.cc:434] [0] SetState old=1, new=2.
I0413 08:35:26.201972    4793 single_queue_dma_scheduler.cc:69] Request[0]: Submitted
I0413 08:35:26.202004    4793 single_tpu_request.cc:357] [0] NotifyRequestActive()
I0413 08:35:26.202027    4793 single_tpu_request.cc:434] [0] SetState old=2, new=3.
I0413 08:35:26.202054    4793 single_queue_dma_scheduler.cc:119] Request[0]: Scheduling DMA[0]
I0413 08:35:26.202084    4793 kernel_registers.cc:173] Write: offset = 0x00000000000485a8, value = 0x0000000000000001
I0413 08:35:26.202119    4793 single_tpu_request.cc:59] [1] Request constructed.
I0413 08:35:26.202146    4793 single_tpu_request.cc:89] Adding input "input" with 150528 bytes.
I0413 08:35:26.202172    4793 single_tpu_request.cc:157] Adding output "MobilenetV1/Predictions/Reshape_1" with 1001 bytes.
I0413 08:35:26.202252    4793 instruction_buffers.cc:32] InstructionBuffers created.
I0413 08:35:26.202273    4793 package_registry.cc:535] Created new instruction buffers.
I0413 08:35:26.202292    4793 device_buffer_mapper.cc:100] Mapped scratch : (nil) -> 0x0000000000000000, 0 bytes.
I0413 08:35:26.202347    4793 kernel_mmu_mapper.cc:124] MmuMapper#Map() : 0000aaaae2750000 -> 8000000000840000 (37 pages).
I0413 08:35:26.202373    4793 mmio_address_space.cc:47] MapMemory() page-aligned : device_address = 0x8000000000840000
I0413 08:35:26.202396    4793 device_buffer_mapper.cc:59] Mapped input "input" : 0xaaaae2750200 -> 0x8000000000840200, 150528 bytes.
I0413 08:35:26.202426    4793 kernel_mmu_mapper.cc:124] MmuMapper#Map() : 0000aaaae277e000 -> 8000000000802000 (1 pages).
I0413 08:35:26.202450    4793 mmio_address_space.cc:47] MapMemory() page-aligned : device_address = 0x8000000000802000
I0413 08:35:26.202474    4793 device_buffer_mapper.cc:81] Mapped output "MobilenetV1/Predictions/Reshape_1" : 0xaaaae277e000 -> 0x8000000000802000, 1008 bytes.
I0413 08:35:26.202499    4793 single_tpu_request.cc:324] MapDataBuffers() done.
I0413 08:35:26.202522    4793 executable_util.cc:78] Linking input[0]: 0x8000000000840200
I0413 08:35:26.202548    4793 executable_util.cc:78] Linking MobilenetV1/Predictions/Reshape_1[0]: 0x8000000000802000
I0413 08:35:26.202590    4793 kernel_mmu_mapper.cc:124] MmuMapper#Map() : 0000aaaae2782000 -> 8000000000820000 (19 pages).
I0413 08:35:26.202615    4793 mmio_address_space.cc:47] MapMemory() page-aligned : device_address = 0x8000000000820000
I0413 08:35:26.202638    4793 device_buffer_mapper.cc:118] Mapped instructions[0] : 0xaaaae2782000 -> 0x8000000000820000, 77776 bytes.
I0413 08:35:26.202663    4793 single_tpu_request.cc:340] MapInstructionBuffers() done.
I0413 08:35:26.202683    4793 single_tpu_request.cc:434] [1] SetState old=0, new=1.
I0413 08:35:26.202705    4793 single_tpu_request.cc:349] [1] NotifyRequestSubmitted()
I0413 08:35:26.202649    4794 kernel_event_handler.cc:70] event_fd=7. Monitor thread got num_events=1.
I0413 08:35:26.202787    4793 single_tpu_request.cc:434] [1] SetState old=1, new=2.
I0413 08:35:26.202874    4794 kernel_registers.cc:173] Write: offset = 0x00000000000485c8, value = 0x0000000000000000
I0413 08:35:26.202877    4793 single_queue_dma_scheduler.cc:69] Request[1]: Submitted
I0413 08:35:26.202961    4793 single_tpu_request.cc:357] [1] NotifyRequestActive()
I0413 08:35:26.203019    4793 single_tpu_request.cc:434] [1] SetState old=2, new=3.
I0413 08:35:26.203087    4793 single_queue_dma_scheduler.cc:119] Request[1]: Scheduling DMA[0]
I0413 08:35:26.203159    4793 kernel_registers.cc:173] Write: offset = 0x00000000000485a8, value = 0x0000000000000002
I0413 08:35:26.203161    4794 single_queue_dma_scheduler.cc:141] Completing DMA[0]
I0413 08:35:26.203386    4794 host_queue.h:387] Completed 1 elements.
I0413 08:35:26.213043    4795 kernel_event_handler.cc:70] event_fd=11. Monitor thread got num_events=1.
I0413 08:35:26.213157    4795 kernel_registers.cc:194] Read: offset = 0x00000000000486d0, value: = 0x0000000000000001
I0413 08:35:26.213198    4795 kernel_registers.cc:173] Write: offset = 0x00000000000486a8, value = 0x000000000000000e
I0413 08:35:26.213228    4795 single_tpu_request.cc:366] [0] NotifyCompletion()
I0413 08:35:26.213257    4795 kernel_mmu_mapper.cc:150] MmuMaper#Unmap() : 0000aaaae2778000 -> 8000000000800000 (2 pages).
I0413 08:35:26.213286    4795 mmio_address_space.cc:74] UnmapMemory() page-aligned : device_address = 0x8000000000800000, num_pages = 2
I0413 08:35:26.213313    4795 package_registry.cc:546] Returned instruction buffers back to executable reference
I0413 08:35:26.213335    4795 single_tpu_request.cc:434] [0] SetState old=3, new=4.
I0413 08:35:26.213358    4795 single_queue_dma_scheduler.cc:221] Request[0]: Completed
I0413 08:35:26.213380    4795 single_tpu_request.cc:73] [0] Request destroyed.
I0413 08:35:26.213160    4794 kernel_event_handler.cc:70] event_fd=7. Monitor thread got num_events=1.
I0413 08:35:26.213594    4794 kernel_registers.cc:173] Write: offset = 0x00000000000485c8, value = 0x0000000000000000
I0413 08:35:26.213615    4794 single_queue_dma_scheduler.cc:141] Completing DMA[0]
I0413 08:35:26.213635    4794 host_queue.h:387] Completed 1 elements.
I0413 08:35:26.215071    4795 kernel_event_handler.cc:70] event_fd=11. Monitor thread got num_events=1.
I0413 08:35:26.215150    4795 kernel_registers.cc:194] Read: offset = 0x00000000000486d0, value: = 0x0000000000000002
I0413 08:35:26.215225    4795 kernel_registers.cc:173] Write: offset = 0x00000000000486a8, value = 0x000000000000000e
I0413 08:35:26.215301    4795 single_tpu_request.cc:366] [1] NotifyCompletion()
I0413 08:35:26.215384    4795 kernel_mmu_mapper.cc:150] MmuMaper#Unmap() : 0000aaaae2782000 -> 8000000000820000 (19 pages).
I0413 08:35:26.215459    4795 mmio_address_space.cc:74] UnmapMemory() page-aligned : device_address = 0x8000000000820000, num_pages = 19
I0413 08:35:26.215557    4795 kernel_mmu_mapper.cc:150] MmuMaper#Unmap() : 0000aaaae2750000 -> 8000000000840000 (37 pages).
I0413 08:35:26.215631    4795 mmio_address_space.cc:74] UnmapMemory() page-aligned : device_address = 0x8000000000840000, num_pages = 37
I0413 08:35:26.215708    4795 kernel_mmu_mapper.cc:150] MmuMaper#Unmap() : 0000aaaae277e000 -> 8000000000802000 (1 pages).
I0413 08:35:26.215781    4795 mmio_address_space.cc:74] UnmapMemory() page-aligned : device_address = 0x8000000000802000, num_pages = 1
I0413 08:35:26.215860    4795 package_registry.cc:546] Returned instruction buffers back to executable reference
I0413 08:35:26.215993    4795 single_tpu_request.cc:434] [1] SetState old=3, new=4.
I0413 08:35:26.216066    4795 single_queue_dma_scheduler.cc:221] Request[1]: Completed
I0413 08:35:26.216137    4795 single_tpu_request.cc:73] [1] Request destroyed.
invoked 
average time: 16.521 ms 
0.796078: 653 653:military uniform
0.0901961: 907 907:Windsor tie
0.0156863: 458 458:bow tie, bow-tie, bowtie
0.0117647: 466 466:bulletproof vest
0.00392157: 922 922:book jacket, dust cover, dust jacket, dust wrapper
I0413 08:35:26.218554    4793 kernel_mmu_mapper.cc:150] MmuMaper#Unmap() : 0000ffff8fdb3000 -> 8000000000000000 (1079 pages).
I0413 08:35:26.218633    4793 mmio_address_space.cc:74] UnmapMemory() page-aligned : device_address = 0x8000000000000000, num_pages = 1079
I0413 08:35:26.218673    4793 instruction_buffers.cc:37] InstructionBuffers destroyed.
I0413 08:35:26.218690    4793 instruction_buffers.cc:37] InstructionBuffers destroyed.
```

See [here](https://github.com/freedomtan/edge_tpu_python_scripts) for scripts using [Python API](https://coral.withgoogle.com/docs/edgetpu/api-intro/)
