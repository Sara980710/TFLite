#include <iostream>
#include <chrono>
#include "Detector.h"

Detector::Detector(const char *model_path, bool gpu, int threads, bool verbose){

  // Read model 
  if (verbose) {
      std::cout<<"-***-Reading model...\n";
  }
  
  model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (model == nullptr) {
    throw std::runtime_error("The model was not able to build to tflite::FlatBufferModel");
  }
  
  if (verbose) {
      std::cout<<"-***-Initiating interpreter...\n";
  }
  
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter == nullptr){
    throw std::runtime_error("Failed to initiate the interpreter");
  }

  if (verbose) {
    std::cout<<"-***-Allocate tensors...\n";
  }
  
  if (interpreter->AllocateTensors() != kTfLiteOk){
    throw std::runtime_error("Failed to allocate tensor");
  }

  // Configure the interpreter
  interpreter->SetNumThreads(threads);
  
  // Get info about model 
  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  auto input_type = interpreter->tensor(input)->type;
  int input_batch = dims->data[0];
  int input_height = dims->data[1];
  int input_width = dims->data[2];
  int input_channels = dims->data[3];

  if (verbose) {
    std::cout<<"-***-Get model info...\n" \
    << "-*i*-- tensors size: " << interpreter->tensors_size()<<"\n";
    std::cout << "-*i*-- nr of operations: " << interpreter->nodes_size()<<"\n";
    std::cout<<"-*i*-- Input height: "<<input_height<<"\n";
    std::cout<<"-*i*-- Input width: "<<input_width<<"\n";
    std::cout<<"-*i*-- Input Nr channels: "<<input_channels<<"\n";
    std::cout<<"-*i*-- Input type: "<<input_type<<std::endl;
  }
};

Detector::~Detector(){};

void Detector::load_image(const char *image_path, int desiredPrecision, bool normalize, bool verbose){
  // Load image 
  if (verbose) std::cout<<"-***-Loading image...\n";
  
  auto frame = cv::imread(image_path);
  if (frame.empty())
  {
      throw std::runtime_error("Failed to load image");
      exit(-1);
  }
  image = frame;
  int input = interpreter->inputs()[0];
  if (image.size().width != interpreter->tensor(input)->dims->data[2] || image.size().height != interpreter->tensor(input)->dims->data[1]) {
      throw std::runtime_error("Image dimensions does not match teh input dimensions on model");
      exit(-1);
  }

  // Check input type
  if (verbose) {
      std::cout<<"-***-Creating tensors...\n";
  }
  
  switch (interpreter->tensor(input)->type)
    {
    case kTfLiteFloat32:
      if (desiredPrecision == 0) {
        if (verbose) std::cout<<"-*i*-- Input precision: "<<"Float32"<<std::endl;
      } 
      else if (desiredPrecision == 1) {
        interpreter->SetAllowFp16PrecisionForFp32(true);
        if (verbose) std::cout<<"-*i*-- Input type: "<<"Float16"<<std::endl;
      } 
      else {
        fprintf(stderr, "cannot handle precision given in the arguments\n");
      }
      if (normalize) {
        image.convertTo(image, CV_32F, 1.0 / 255, 0);
      } else {
        image.convertTo(image, CV_32F, 1.0, 0);
      }

      break;
    case kTfLiteUInt8:
        if (verbose) std::cout<<"-*i*-- Input type: "<<"int8"<<std::endl;
        break;
    default:
        fprintf(stderr, "cannot handle input type\n");
        exit(-1);
    }

    memcpy(interpreter->typed_tensor<float>(0), image.data, image.total() * image.elemSize());

    // Normalize to plot with right values
    if (!normalize) {
      image.convertTo(image, CV_32F, 1.0/255, 0);
    }
};

void Detector::detect(bool verbose) {
  // Warmup to make sure cache are warm and any JIT compiling is in the way has run/tuned
  if (verbose) std::cout<<"-***-Warm up...\n";
  if (interpreter->Invoke() != kTfLiteOk){
    throw std::runtime_error("Failed to run model");
  }
  
  if (verbose) std::cout<<"-***-Run...\n";
  auto t1 = std::chrono::high_resolution_clock::now();
  interpreter->Invoke();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  if (verbose) std::cout << "-*i*-- Inference time: "<< duration << " us" << std::endl;
}


