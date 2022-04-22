#include <iostream>
#include "Detector.h"


int main(int argc, char **argv) {
  if (argc != 5) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -iterations for stable measures\n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input");
  }
  const char *modelFileName = argv[1];
  const int iterations = std::stoi(argv[2]);
  const int desiredPrecision = std::stoi(argv[3]);
  const char *imageFileName = argv[4];

  int threads = 1;
  bool verbose = false;
  bool gpu = false;
  bool normalize = true;
  bool time = true;
  bool memory = false;

  Detector detector(modelFileName, gpu, threads, verbose);
  auto t1 = std::chrono::high_resolution_clock::now();
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  t1 = std::chrono::high_resolution_clock::now();
  detector.tile_image(verbose);
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

  // Run
  t1 = std::chrono::high_resolution_clock::now();
  detector.load_input(true);
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  t1 = std::chrono::high_resolution_clock::now();
  detector.detect(verbose);
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();


  // Process output
  float* output = detector.get_output(verbose);

  std::cout<<"-***-Done! \n";
  return 0;
}

