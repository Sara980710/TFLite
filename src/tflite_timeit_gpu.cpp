#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "TF/tensorflow/lite/tools/gen_op_registration.h"
#include "TF/tensorflow/lite/delegates/gpu/delegate.h"

#include <chrono>
#include <string>
#include <numeric>
#include <iostream>


int main(int argc, char **argv) {
  if (argc != 4) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -iterations for stable measures\n            -desired precision (0:f32, 1:f16, 2:int8)");
  }
  const char *modelFileName = argv[1];
  const int iterations = std::stoi(argv[2]);
  const int desiredPrecision = std::stoi(argv[3]);

  std::cout<<"\n ---------------------------------------\n";
  std::cout<<"Reading model...\n";
  auto model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
  if (model == nullptr) {
    throw std::runtime_error("The model was not able to build to tflite::FlatBufferModel");
  }

  std::cout<<"Initiating interpreter...\n";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter == nullptr){
    throw std::runtime_error("Failed to initiate the interpreter");
  }

  // Enable use of the GPU delegate, remove below lines to get cpu
  auto* delegate = TfLiteGpuDelegateV2Create(nullptr);
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    std::cout << "Fail" << std::endl;
    return -1;
  }

  std::cout<<"Allocate tensors...\n";
  if (interpreter->AllocateTensors() != kTfLiteOk){
    throw std::runtime_error("Failed to allocate tensor");
  }

  // Determine precision
  int input = interpreter->inputs()[0];
  switch (interpreter->tensor(input)->type)
    {
    case kTfLiteFloat32:
      if (desiredPrecision == 0) {
        std::cout<<"precision: "<<"Float32"<<std::endl;
      } 
      else if (desiredPrecision == 1) {
        interpreter->SetAllowFp16PrecisionForFp32(true);
        std::cout<<"precision: "<<"Float16"<<std::endl;
      } 
      else {
        fprintf(stderr, "cannot handle precision given in the arguments\n");
      }
      break;
    case kTfLiteUInt8:
        std::cout<<"precision: "<<"int8"<<std::endl;
        break;
    default:
        fprintf(stderr, "cannot handle input type\n");
        exit(-1);
    }
  
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  std::cout<<"Warm up...\n";
  interpreter->Invoke();

  // run single invoation and measure time
  std::vector<float> timeMeasures;
  std::cout<<"Running " << iterations << " iterations...\n";
  for (int i=0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    interpreter->Invoke();
    auto t2 = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    timeMeasures.push_back(duration);
  }

  auto const count = static_cast<float>(timeMeasures.size());
  float duration = std::accumulate(timeMeasures.begin(), timeMeasures.end(), 0) / count;
  
  std::cout << duration << " us" << std::endl;
  std::cout<<"Done! \n";
  std::cout<<"---------------------------------------\n"<<std::endl;
  return 0;
}

