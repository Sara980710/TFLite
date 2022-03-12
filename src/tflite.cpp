#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <iostream>
#include <chrono>

int main() {
  // Example of simple model from the tensorflow repository
  // Can be replaced for quantized models but as of this writing they perform worse on X86-platforms
  std::cout<<"Reading model...";
  auto model = tflite::FlatBufferModel::BuildFromFile("/home/sara/Desktop/Master-thesis/TFLite/models/classification/example_mobilenet_v1_1.0_224_quant.tflite");
  std::cout<<"Done! \n";

  std::cout<<"Creating vaiables...";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  std::cout<<"Done! \n";

  std::cout<<"Allocate tensors...";
  interpreter->AllocateTensors();
  std::cout<<"Done! \n";
  
  // input/output vector, no content specified in this case
  std::cout<<"Creating tensors...";
  float* input = interpreter->typed_input_tensor<float>(0);
  float* output = interpreter->typed_output_tensor<float>(0);
  std::cout<<"Done! \n";
  
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  std::cout<<"Warm up...";
  interpreter->Invoke();
  std::cout<<"Done! \n";

  // run single invoation and measure time
  std::cout<<"Run...";
  auto t1 = std::chrono::high_resolution_clock::now();
  interpreter->Invoke();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  std::cout << duration << " us" << std::endl;
  std::cout<<"Done! \n";
  return 0;
}

