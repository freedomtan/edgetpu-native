// Example to run two models with two Edge TPUs using two threads.
// It depends only on tflite and edgetpu.h
//
// Example usage:
// 1. Create directory /tmp/edgetpu_cpp_example
// 2. wget -O /tmp/edgetpu_cpp_example/inat_bird_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
// 3. wget -O /tmp/edgetpu_cpp_example/inat_plant_edgetpu.tflite \
//      http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite
// 4. wget -O /tmp/edgetpu_cpp_example/bird.jpg \
//      https://farm3.staticflickr.com/4003/4510152748_b43c1da3e6_o.jpg
// 5. wget -O /tmp/edgetpu_cpp_example/plant.jpg \
//      https://c2.staticflickr.com/1/62/184682050_db90d84573_o.jpg
// 6. cd /tmp/edgetpu_cpp_example && mogrify -format bmp *.jpg
// 7. Build and run `two_models_two_tpus_threaded`
//
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT

#include "edgetpu.h"
#include "edgetpu/cpp/examples/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

int main() {
  // Modify the follow according to try different models.
  const int num_inferences = 2000;
  const std::string bird_model_path =
      "/tmp/edgetpu_cpp_example/inat_bird_edgetpu.tflite";
  const std::string plant_model_path =
      "/tmp/edgetpu_cpp_example/inat_plant_edgetpu.tflite";
  // Both models take input size of 224x224x3.
  const int height = 224, width = 224, channels = 3;
  const coral::ImageDims model_input_size = {height, width, channels};
  const std::string bird_image_path = "/tmp/edgetpu_cpp_example/bird.bmp";
  const std::string plant_image_path = "/tmp/edgetpu_cpp_example/plant.bmp";

  const auto& available_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  if (available_tpus.size() < 2) {
    std::cerr << "This example requires two Edge TPUs to run." << std::endl;
    return 0;
  }

  auto thread_job =
      [&model_input_size](
          const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& tpu,
          const std::string& model_path, const std::string& image_path) {
        const auto& tid = std::this_thread::get_id();
        std::cout << "Thread: " << tid << " Using model: " << model_path
                  << " Running " << num_inferences << " inferences."
                  << std::endl;
        std::unordered_map<std::string, std::string> options = {
            {"Usb.MaxBulkInQueueLength", "8"},
        };
        auto tpu_context =
            edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext(
                tpu.type, tpu.path, options);
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (model == nullptr) {
          std::cerr << "Fail to build FlatBufferModel from file: " << model_path
                    << std::endl;
          std::abort();
        }
        std::vector<uint8_t> input =
            coral::GetInputFromBmpImage(image_path, model_input_size);
        std::unique_ptr<tflite::Interpreter> interpreter =
            coral::BuildEdgeTpuInterpreter(*model, tpu_context.get());
        std::cout << "Thread: " << tid << " Interpreter was built."
                  << std::endl;
        for (int i = 0; i < num_inferences; ++i) {
          coral::RunInference(input, interpreter.get());
        }
        // Print inference result.
        const auto& result = coral::RunInference(input, interpreter.get());
        auto it_a = std::max_element(result.begin(), result.end());
        std::cout << "Thread: " << tid
                  << " printing analysis result. Max value index: "
                  << std::distance(result.begin(), it_a) << " value: " << *it_a
                  << std::endl;
      };

  const auto& start_time = std::chrono::steady_clock::now();
  std::thread bird_model_thread(thread_job, available_tpus[0], bird_model_path,
                                bird_image_path);
  std::thread plant_model_thread(thread_job, available_tpus[1],
                                 plant_model_path, plant_image_path);
  bird_model_thread.join();
  plant_model_thread.join();
  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;
  std::cout << "Using two Edge TPUs, # inferences: " << num_inferences
            << " costs: " << time_span.count() << " seconds." << std::endl;

  return 0;
}
