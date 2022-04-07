#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc != 4) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -path to image input \n            -path to label");
  }
  const char *modelFileName = argv[1];
  const char *imageFileName = argv[2];
  const char *lableFileName = argv[3];

  // Read model 
  std::cout<<"-***-Reading model...\n";
  auto model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
  if (model == nullptr) {
    throw std::runtime_error("The model was not able to build to tflite::FlatBufferModel");
  }

  std::cout<<"-***-Initiating interpreter...\n";
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter == nullptr){
    throw std::runtime_error("Failed to initiate the interpreter");
  }

  std::cout<<"-***-Allocate tensors...\n";
  if (interpreter->AllocateTensors() != kTfLiteOk){
    throw std::runtime_error("Failed to allocate tensor");
  }

  // Configure the interpreter
  interpreter->SetAllowFp16PrecisionForFp32(true);
  interpreter->SetNumThreads(1);
  
  // Get info about model 

  std::cout<<"-***-Get model info...\n";
  std::cout << "-*i*-- tensors size: " << interpreter->tensors_size()<<"\n";
  std::cout << "-*i*-- nr of operations: " << interpreter->nodes_size()<<"\n";

  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  std::cout<<"-*i*-- Input height: "<<wanted_height<<"\n";
  std::cout<<"-*i*-- Input width: "<<wanted_width<<"\n";
  std::cout<<"-*i*-- Input Nr channels: "<<wanted_channels<<"\n";

  // Load image 
  std::cout<<"-***-Loading image...\n";
  cv::Mat image;
  auto frame = cv::imread(imageFileName);
  if (frame.empty())
  {
      throw std::runtime_error("Failed to load image");
      exit(-1);
  }
  image = frame;
  image.convertTo(image, -1, 1.0 / 255, 0);

  std::cout<<"-***-Creating tensors...\n";

  memcpy(interpreter->typed_tensor<float>(0), image.data, image.total() * image.elemSize());

  // Run
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  std::cout<<"-***-Warm up...\n";
  if (interpreter->Invoke() != kTfLiteOk){
    throw std::runtime_error("Failed to run model");
  }
  

  std::cout<<"-***-Run...\n";
  auto t1 = std::chrono::high_resolution_clock::now();
  interpreter->Invoke();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  std::cout << "-*i*-- Inference time: "<< duration << " us" << std::endl;

  // Get output


  int* out_dims = interpreter->output_tensor(0)->dims->data;
  int batch_size = interpreter->output_tensor(0)->dims->data[0];
  int num_dets = interpreter->output_tensor(0)->dims->data[1];
  int det_size = interpreter->output_tensor(0)->dims->data[2];

  std::cout<<"-*i*- Batch size: "<<batch_size<<"\n";
  std::cout<<"-*i*- Number of detections: "<<num_dets<<"\n";
  std::cout<<"-*i*- Detection size: "<<det_size<<"\n";

  bool printall = true;
  float* coords_t = interpreter->typed_output_tensor<float>(0);
  for (int i = 0; i < det_size*num_dets; i=i+det_size) {
    float conf = coords_t[i+4]*coords_t[i+5];
      if (printall and conf > 1) {
        std::cout<<"-*i*----Detection: "<<i/det_size<<"\n";
        std::cout<<"-*i*-----x: "<<coords_t[i]<<"\n";
        std::cout<<"-*i*-----y: "<<coords_t[i+1]<<"\n";
        std::cout<<"-*i*-----w: "<<coords_t[i+2]<<"\n";
        std::cout<<"-*i*-----h: "<<coords_t[i+3]<<"\n";
        std::cout<<"-*i*-----confidence: "<<conf<<"\n";
        std::cout<<"-*i*-----class: "<<coords_t[i+5]<<"\n";
      }
      
  }


  //cv::imshow("img", image);
  //cv::waitKey(0);

  std::cout<<"-***-Done! \n";
  return 0;
}

