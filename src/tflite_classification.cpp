#include <iostream>
#include "Detector.h"

int main(int argc, char **argv) {
  if (argc != 4) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input");
  }
  const char *modelFileName = argv[1];
  int desiredPrecision = std::stoi(argv[2]);
  const char *imageFileName = argv[3];

  int threads = 1;
  bool verbose = true;
  bool gpu = false;
  bool normalize = false;
  
  Detector detector(modelFileName, gpu, threads, verbose);
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);

  // Run
  detector.detect(verbose);


  // Process output
  float* output = detector.interpreter->typed_output_tensor<float>(0);

  std::cout<<"-*i*-----boat: "<<output[0]<<"\n";
  std::cout<<"-*i*-----no boat: "<<output[1]<<"\n";

  std::string output_txt = "BOAT: " + std::to_string(output[0]) + ", NO BOAT: " + std::to_string(output[1]);
  cv::putText(detector.image, output_txt, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  cv::imshow("output", detector.image);
  cv::waitKey(0);

  std::cout<<"Done! \n";
  return 0;
}

