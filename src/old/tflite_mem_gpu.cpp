#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <chrono>
#include <string>
#include <numeric>
#include <iostream>
#include <memory>
#include <sys/resource.h>
#include <math.h>

struct AllocationMetrics
{ 
  uint32_t StaringUsage = 0;
  uint32_t TotalAllocated = 0;
  uint32_t TotalFreed = 0;
  uint32_t MaxUsage = 0;

  uint32_t GetCurrentAllocation() { return TotalAllocated-TotalFreed;}
};

static AllocationMetrics s_AllocationMetrics;

void* operator new(size_t size) {
  s_AllocationMetrics.TotalAllocated += size;
  if (s_AllocationMetrics.GetCurrentAllocation() > s_AllocationMetrics.MaxUsage) {
    s_AllocationMetrics.MaxUsage = s_AllocationMetrics.GetCurrentAllocation();
  }

  return malloc(size);
}

void operator delete(void* memory, size_t size) {
  s_AllocationMetrics.TotalFreed += size;

  free(memory);
}


int main(int argc, char **argv) {
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  s_AllocationMetrics.StaringUsage = res.ru_maxrss;

  if (argc != 3) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -desired precision (0:f32, 1:f16, 2:int8)");
  }
  const char *modelFileName = argv[1];
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
  std::cout<<"Activating GPU...\n";
  
  // Enable use of the GPU delegate, remove below lines to get cpu
  // After TFlite >=2.6, initiate the gpu options
  TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();

  auto* delegate = TfLiteGpuDelegateV2Create(&gpu_options);
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
  float* input_tensor_float;
  uint8_t* input_tensor_int;

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
      input_tensor_float = interpreter->typed_input_tensor<float>(0);
      break;
    case kTfLiteUInt8:
        std::cout<<"precision: "<<"int8"<<std::endl;
        input_tensor_int = interpreter->typed_input_tensor<uint8_t>(0);
        break;
    default:
        fprintf(stderr, "cannot handle input type\n");
        exit(-1);
    }
  
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  std::cout<<"Warm up...\n";
  interpreter->Invoke();

  // run single invoation and measure time
  for (int i=0; i < 1; i++) {
  interpreter->Invoke();
  }
  std::cout<<"Max memory allocated: " << s_AllocationMetrics.MaxUsage*pow(10,-6)<< " MB" << std::endl;
  std::cout<<"Memory baseline: " << s_AllocationMetrics.StaringUsage<< " Bytes" << std::endl;
  std::cout<<"Memory used soleny by this process: " << (s_AllocationMetrics.MaxUsage + s_AllocationMetrics.StaringUsage) *pow(10,-6)<< " MB" << std::endl;
  return 0;
}

