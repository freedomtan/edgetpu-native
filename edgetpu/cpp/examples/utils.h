#ifndef EDGETPU_CPP_EXAMPLES_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_UTILS_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace coral {

typedef std::array<int, 3> ImageDims;

// Reads BMP image from file and resizes to `target_dims`. It uses bilinear
// interpolation method to resize.
std::vector<uint8_t> GetInputFromBmpImage(const std::string& image_path,
                                          const ImageDims& target_dims);

// Builds tflite Interpreter capable of running Edge TPU model.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);

// Runs inference using given `interpreter`
std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                tflite::Interpreter* interpreter);

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_UTILS_H_
