#include <iostream>
#include "Detector.h"


int main(int argc, char **argv) {
  if (argc != 6) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input\n            -Number of threads\n            -device(gpu or cpu)");
  }
  const char *modelFileName = argv[1];
  const int desiredPrecision = std::stoi(argv[2]);
  const char *imageFileName = argv[3];
  const int threads = std::stoi(argv[4]);

  bool gpu = false;
  if (std::stoi(argv[5]) == 1) {
    gpu = true;
  }
  
  bool verbose = false;
  bool normalize = true;

  Detector detector(modelFileName, gpu, threads, verbose);

  // Pre process
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);
  detector.tile_image(verbose);

  // Run
  while (detector.currentTile != -1) {
    detector.load_input(verbose);
    detector.detect(verbose);
    float* output = detector.get_output(verbose);
  }

  std::cout<<"-***-Done! \n";
  return 0;
}

