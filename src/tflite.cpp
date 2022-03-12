#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <iostream>
#include <chrono>

int main() {
  // Example of simple model from the tensorflow repository
  // Can be replaced for quantized models but as of this writing they perform worse on X86-platforms
  auto model = tflite::FlatBufferModel::BuildFromFile("/home/sara/Desktop/Master-thesis/best-fp 16.tflite");

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  interpreter->AllocateTensors();
  
  // input/output vector, no content specified in this case
  float* input = interpreter->typed_input_tensor<float>(0);
  float* output = interpreter->typed_output_tensor<float>(0);
  
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  interpreter->Invoke();

  // run single invoation and measure time
  auto t1 = std::chrono::high_resolution_clock::now();
  interpreter->Invoke();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  std::cout << duration << " us" << std::endl;
  return 0;
}

