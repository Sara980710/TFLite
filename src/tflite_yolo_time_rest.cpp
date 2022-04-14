#include <iostream>
#include "Detector.h"

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
  if (argc != 4) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input\n            -gpu (1:true, 0:false)\n            -nr threads");
  }
  const char *modelFileName = argv[1];
  int desiredPrecision = std::stoi(argv[2]);
  const char *imageFileName = argv[3];
  if (std::stoi(argv[4]) == 1) {
    bool gpu = true;
  } else {
    bool gpu = false;
  }
  int threads = std::stoi(argv[5]);
  
  bool verbose = false;
  bool normalize = true;

  auto t1 = std::chrono::high_resolution_clock::now();
  Detector detector(modelFileName, gpu, threads, verbose);
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);
  auto t2 = std::chrono::high_resolution_clock::now();
  float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  
  std::cout << std::fixed;
  std::cout << "Preprocess time: " << duration << " us" << std::endl;
  std::cout<<"Done! \n";
  std::cout<<"---------------------------------------\n"<<std::endl;

  // Run
  detector.detect(verbose);

  // Time post-processing
  std::vector<float> timeMeasures;
  std::cout<<"Running " << iterations << " iterations...\n";
  for (int i=0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();

    int* out_dims = detector.interpreter->output_tensor(0)->dims->data;
    int batch_size = out_dims[0];
    int num_dets = out_dims[1];
    int det_size = out_dims[2];

    std::cout<<"-*i*- Batch size: "<<batch_size<<"\n";
    std::cout<<"-*i*- Number of detections: "<<num_dets<<"\n";
    std::cout<<"-*i*- Detection size: "<<det_size<<"\n";


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
    auto t2 = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    timeMeasures.push_back(duration);
  }

  auto const count = static_cast<float>(timeMeasures.size());
  float duration = std::accumulate(timeMeasures.begin(), timeMeasures.end(), 0) / count;
  
  std::cout << std::fixed;
  std::cout << "Average post-process time 1 invoke: " << duration << " us" << std::endl;
  std::cout<<"Done! \n";
  std::cout<<"---------------------------------------\n"<<std::endl;

  return 0;
}

