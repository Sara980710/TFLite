#include <iostream>
#include "Detector.h"


int main(int argc, char **argv) {
  if (argc != 7) {
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -iterations for stable measures\n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input\n            -Number of threads\n            -device(gpu or cpu)");
  }
  const char *modelFileName = argv[1];
  const int iterations = std::stoi(argv[2]);
  const int desiredPrecision = std::stoi(argv[3]);
  const char *imageFileName = argv[4];
  const int threads = std::stoi(argv[5]);

  bool gpu = false;
  if (argv[6] == "gpu") {
    bool gpu = true;
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

