#include <iostream>
#include <chrono>
#include "Detector.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

Detector::Detector(const char *model_path, bool gpu, int threads, bool verbose){
  // Read model 
  if (verbose) std::cout<<"Reading model...\n";
  m_model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (m_model == nullptr) {
    throw std::runtime_error("The model was not able to build to tflite::FlatBufferModel");
  }
  
  if (verbose) std::cout<<"Initiating interpreter...\n";
  tflite::InterpreterBuilder(*m_model, m_resolver)(&m_interpreter);
  if (m_interpreter == nullptr){
    throw std::runtime_error("Failed to initiate the interpreter");
  }

  // Set GPU
  if (gpu) {
    if (verbose) std::cout<<"Activating GPU...\n";
    
    // Enable use of the GPU delegate, remove below lines to get cpu
    // After TFlite >=2.6, initiate the gpu options
    TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();

    m_delegate = TfLiteGpuDelegateV2Create(&gpu_options);
    if (m_interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk) {
      throw std::runtime_error("Failed to create GPU deligate");
      exit(-1);
    }
    m_gpu = true;
  } else {
    m_gpu = false;
  }

  // Allocating tensors
  if (verbose) std::cout<<"Allocate tensors...\n";
  if (m_interpreter->AllocateTensors() != kTfLiteOk){
    throw std::runtime_error("Failed to allocate tensor");
  }

  // Configure the interpreter
  m_interpreter->SetNumThreads(threads);

  int input = m_interpreter->inputs()[0];
  inDims = m_interpreter->tensor(input)->dims->data;
  outDims = m_interpreter->output_tensor(0)->dims->data;

  if (verbose) {
    // Get info about model 
    std::cout<<"-***-Get model info...\n" \
    << "-*i*-- tensors size: " << m_interpreter->tensors_size()<<"\n";
    std::cout << "-*i*-- nr of operations: " << m_interpreter->nodes_size()<<"\n";
    std::cout<<"-*i*-- Input batch size: "<<inDims[0]<<"\n";
    std::cout<<"-*i*-- Input height: "<<inDims[1]<<"\n";
    std::cout<<"-*i*-- Input width: "<<inDims[2]<<"\n";
    std::cout<<"-*i*-- Input Nr channels: "<<inDims[3]<<"\n";
    std::cout<<"-*i*-- Input type: "<<m_interpreter->tensor(input)->type<<std::endl;
  }

  currentTile = 0;
};

Detector::~Detector(){if (m_gpu) {TfLiteGpuDelegateV2Delete(m_delegate);}};

void Detector::load_image(const char *image_path, int desiredPrecision, bool normalize, bool verbose){
  // Load image 
  if (verbose) std::cout<<"Loading image...\n";

  image = cv::imread(image_path);
  if (image.empty())
  {
      throw std::runtime_error("Failed to load image");
      exit(-1);
  }

  int input = m_interpreter->inputs()[0];

  // Check input type
  if (verbose) std::cout<<"Creating tensors...\n";

  switch (m_interpreter->tensor(input)->type)
    {
    case kTfLiteFloat32:
      if (desiredPrecision == 0) {image.convertTo(image, CV_32F, 1.0, 0);
        if (verbose) std::cout<<"Input precision: "<<"Float32"<<std::endl;
      } 
      else if (desiredPrecision == 1) {
        m_interpreter->SetAllowFp16PrecisionForFp32(true);
        if (verbose) std::cout<<"Input type: "<<"Float16"<<std::endl;
      } 
      else {
        fprintf(stderr, "cannot handle precision given in the arguments\n");
      }
      if (normalize) {
        // (yolo)
        image.convertTo(image, CV_32F, 1.0 / 255, 0);
      } else {
        // (classification)
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

    // Normalize to plot with right values (classification)
    if (!normalize) {
      image.convertTo(image, CV_32F, 1.0/255, 0);
    }
};

void Detector::tile_image(bool verbose) {
  if (image.size().width < inDims[2] || image.size().height < inDims[1]) {
    fprintf(stderr, "Cannot handle images smaller than the input size\n");
    exit(-1);
  }

  // Calculate how to tile image
  for (int x=0; x < image.size().width; x += inDims[2]) {
    for (int y=0; y < image.size().height; y += inDims[1]) {
      // Handle edges
      if (x + inDims[2] >= image.size().width && y + inDims[1] >= image.size().height) {
        m_tiles.push_back(cv::Rect(image.size().width - inDims[2], image.size().height - inDims[1], inDims[2], inDims[1]));
      } else {
        if (x + inDims[2] >= image.size().width) {
          m_tiles.push_back(cv::Rect(image.size().width - inDims[2], y, inDims[2], inDims[1]));
        } else if (y + inDims[1] >= image.size().height) {
          m_tiles.push_back(cv::Rect(x, image.size().height - inDims[1], inDims[2], inDims[1]));
        } else {
          m_tiles.push_back(cv::Rect(x, y, inDims[2], inDims[1]));
        }
      }
    }
  }

  if (verbose) {
    std::cout << "Image size: " << image.size() << "\n";
    std::cout << "Input size: [" << inDims[2] << " x " << inDims[1] << "]\n";
    std::cout << "Nr tiles: " << m_tiles.size() << std::endl;
  }
}

void Detector::load_input(bool verbose) {
  if (verbose) std::cout << "Loading input, tile " << currentTile + 1<< "/" << m_tiles.size() << std::endl;
  int input = m_interpreter->inputs()[0];

  switch (m_interpreter->tensor(input)->type)
    {
    case kTfLiteFloat32:
      memcpy(m_interpreter->typed_tensor<float>(0), image(m_tiles[currentTile]).data, inDims[0] * inDims[1] * inDims[2] * inDims[3] * 4); 
      break;
    case kTfLiteUInt8:
        memcpy(m_interpreter->typed_tensor<uint8_t>(0), image(m_tiles[currentTile]).data, inDims[0] * inDims[1] * inDims[2] * inDims[3] * 4);
        break;
    default:
        fprintf(stderr, "cannot handle input type\n");
        exit(-1);
    }
  currentTile ++;
  if (currentTile != -1 && currentTile >= m_tiles.size()) {
    currentTile = -1;
    if (verbose) std::cout << "All tiles done!" << std::endl;
  }
};  

void Detector::detect(bool verbose) {
  if (verbose) std::cout<<"Run...\n";
  m_interpreter->Invoke();
  if (verbose) std::cout << "Inference done!" << std::endl;
}

float* Detector::get_output(bool verbose) {
  if (verbose) std::cout<<"Getting output...\n";
  return m_interpreter->typed_output_tensor<float>(0);
}


