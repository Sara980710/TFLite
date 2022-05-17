#include <iostream>
#include "Detector.h"
#include<chrono>

int main(int argc, char **argv) {
  if (argc != 6) {
    throw std::invalid_argument("Required arguments: \n            "
                                  "-path to TFLite model file \n            "
                                  "-desired precision (0:f32, 1:f16, 2:int8)\n            "
                                  "-Number of threads\n            "
                                  "-device (0:cpu or 1:gpu)\n            "
                                  "-verbose (0:false, 1:true)");
  }
  const char *modelFileName = argv[1];
  const int desiredPrecision = std::stoi(argv[2]);
  const int threads = std::stoi(argv[3]);

  bool gpu = false;
  if (std::stoi(argv[4]) == 1) {
    gpu = true;
  }

  bool verbose = false;
  if (std::stoi(argv[5]) == 1) {
    verbose = true;
  }
  
  bool normalize = true;
  const uint8_t method = 0;

  Detector detector(modelFileName, gpu, threads, verbose);

  // Run
  auto t1 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::high_resolution_clock::now();

  while (std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() < 30000000) {
    detector.load_input(verbose, method);
    detector.detect(verbose);
    float* output = detector.get_output(verbose);
    t2 = std::chrono::high_resolution_clock::now();
  }

  std::cout<<"-***-Done! \n";
  return 0;
}

