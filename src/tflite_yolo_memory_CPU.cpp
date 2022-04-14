#include <iostream>
#include <memory>
#include <sys/resource.h>
#include <math.h>
#include "Detector.h"

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

//Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
std::vector<float> xywh2xyxy(std::vector<float> xywh) {
  std::vector<float> xyxy;
  xyxy.push_back(xywh[0] - xywh[2] / 2); //top left x
  xyxy.push_back(xywh[1] - xywh[3] / 2); //top left y
  xyxy.push_back(xywh[0] + xywh[2] / 2); //bottom right x
  xyxy.push_back(xywh[1] + xywh[3] / 2); //bottom right y

  return xyxy;
}

int main(int argc, char **argv) {
  struct rusage res;
  getrusage(RUSAGE_SELF, &res);
  s_AllocationMetrics.StaringUsage = res.ru_maxrss;

  if (argc != 4) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input");
  }
  const char *modelFileName = argv[1];
  int desiredPrecision = std::stoi(argv[2]);
  const char *imageFileName = argv[3];
  
  int threads = 1;
  bool verbose = false;
  bool gpu = false;
  bool normalize = true;

  Detector detector(modelFileName, gpu, threads, verbose);
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);

  // for realistic memory measure
  cv::Mat dummyImage(10000, 10000, CV_8UC1, Scalar(0, 0, 255));

  // Run
  detector.detect(verbose);

  // Process output
  int* out_dims = detector.interpreter->output_tensor(0)->dims->data;
  int batch_size = out_dims[0];
  int num_dets = out_dims[1];
  int det_size = out_dims[2];

  float* output = detector.interpreter->typed_output_tensor<float>(0);

  int nrBoundingBoxes = 0;
  int imageWidth = detector.image.size().width;
  int imageHeight = detector.image.size().height;

  for (int i = 0; i < det_size*num_dets; i=i+det_size) {
    float conf = output[i+4]*output[i+5];
      if (conf > 0.5) {
        nrBoundingBoxes = nrBoundingBoxes +1;
        std::vector<float> xywh = {output[i], output[i+1], output[i+2], output[i+3]};
        std::vector<float> xyxy = xywh2xyxy(xywh);
        cv::Point p1(xyxy[0]*imageWidth, xyxy[1]*imageHeight);
        cv::Point p2(xyxy[2]*imageWidth, xyxy[3]*imageHeight);
        
        cv::rectangle(detector.image, p1, p2, cv::Scalar(0, 255, 0));
      }
  }

  std::cout<<"Overall " << std::endl;
  std::cout<<"Max memory allocated: " << s_AllocationMetrics.MaxUsage*pow(10,-6)<< " MB" << std::endl;
  std::cout<<"Memory baseline: " << s_AllocationMetrics.StaringUsage<< " Bytes" << std::endl;
  std::cout<<"Memory used soleny by this process: " << (s_AllocationMetrics.MaxUsage + s_AllocationMetrics.StaringUsage) *pow(10,-6)<< " MB" << std::endl;

  std::cout<<"-***-Done! \n";
  return 0;
}

