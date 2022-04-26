#include <iostream>
#include "Detector.h"
#include <numeric>

struct Measures{
  std::vector<float> loadImage;
  std::vector<float> tileImage;
  std::vector<float> loadInput;
  std::vector<float> detect;
  std::vector<float> getOutput;
  std::vector<float> loadInputTotal;
  std::vector<float> detectTotal;
  std::vector<float> getOutputTotal;
};

static Measures s_measures;
  
float getAverage(std::vector<float> timeMeasures) {
    auto const count = static_cast<float>(timeMeasures.size());
    float duration = std::accumulate(timeMeasures.begin(), timeMeasures.end(), 0) / count;
    return duration;
}

void run_once(const char *modelFileName, bool gpu, int threads, bool verbose, const char *imageFileName, int desiredPrecision, bool normalize){
   Detector detector(modelFileName, gpu, threads, verbose);

  // Pre process
  auto t1 = std::chrono::high_resolution_clock::now();
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  s_measures.loadImage.push_back(duration);

  t1 = std::chrono::high_resolution_clock::now();
  detector.tile_image(verbose);
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  s_measures.tileImage.push_back(duration);

  // Run
  std::vector<float> t_loadInput;
  std::vector<float> t_detect;
  std::vector<float> t_getOutput;
  while (detector.currentTile != -1) {
    t1 = std::chrono::high_resolution_clock::now();
    detector.load_input(verbose);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    t_loadInput.push_back(duration);

    t1 = std::chrono::high_resolution_clock::now();
    detector.detect(verbose);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    t_detect.push_back(duration);

    // Process output
    t1 = std::chrono::high_resolution_clock::now();
    float* output = detector.get_output(verbose);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    t_getOutput.push_back(duration);
  }
  s_measures.loadInput.push_back(getAverage(t_loadInput));
  s_measures.detect.push_back(getAverage(t_detect));
  s_measures.getOutput.push_back(getAverage(t_getOutput));
  s_measures.loadInputTotal.push_back(std::accumulate(t_loadInput.begin(), t_loadInput.end(), 0));
  s_measures.detectTotal.push_back(std::accumulate(t_detect.begin(), t_detect.end(), 0));
  s_measures.getOutputTotal.push_back(std::accumulate(t_getOutput.begin(), t_getOutput.end(), 0));
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cout<<argc<<std::endl;
    throw std::invalid_argument("Required arguments: \n            -path to TFLite model file \n            -iterations for stable measures\n            -desired precision (0:f32, 1:f16, 2:int8)\n            -path to image input\n            -Number of threads\n            -device (0:cpu or 1:gpu)");
  }
  const char *modelFileName = argv[1];
  const int iterations = std::stoi(argv[2]);
  const int desiredPrecision = std::stoi(argv[3]);
  const char *imageFileName = argv[4];
  const int threads = std::stoi(argv[5]);

  bool gpu = false;
  if (std::stoi(argv[6]) == 1) {
    gpu = true;
  }
  
  bool verbose = false;
  bool normalize = true;

  std::cout << "Total iterations: " << iterations << std::endl;
  std::cout << "Iteration progress: " << std::flush;
  for (int i=0; i < iterations; i++) {
    std::cout << i + 1<< ", " << std::flush;
    run_once(modelFileName, gpu, threads, verbose, imageFileName, desiredPrecision, normalize);
  }
  std::cout << std::endl;

  float timeLoadImage = getAverage(s_measures.loadImage);
  float timeTileImage = getAverage(s_measures.tileImage);
  float timeLoadInput = getAverage(s_measures.loadInput);
  float timeDetect = getAverage(s_measures.detect);
  float timeGetOutput = getAverage(s_measures.getOutput);
  float timeLoadInputTotal = getAverage(s_measures.loadInputTotal);
  float timeDetectTotal = getAverage(s_measures.detectTotal);
  float timeGetOutputTotal = getAverage(s_measures.getOutputTotal);

  std::cout << std::fixed;

  std::cout << "\nMeasures detail, (average, min, max): \n";
  std::cout << "- Preprocessing: " << timeLoadImage + timeTileImage <<" us\n";
  std::cout << "  - Load image: (" << timeLoadImage << ", " \
                                  << *std::min_element(s_measures.loadImage.begin(), s_measures.loadImage.end()) << ", " \
                                  << *std::max_element(s_measures.loadImage.begin(), s_measures.loadImage.end()) << ") us\n";
  std::cout << "  - Tile image: (" << timeTileImage << ", " \
                                  << *std::min_element(s_measures.tileImage.begin(), s_measures.tileImage.end()) << ", " \
                                  << *std::max_element(s_measures.tileImage.begin(), s_measures.tileImage.end()) << ") us\n";
  std::cout << "- Inference: " << timeLoadInputTotal + timeDetectTotal + timeGetOutputTotal <<" us\n";
  std::cout << "  - Load input total: (" << timeLoadInputTotal << ", " \
                                  << *std::min_element(s_measures.loadInputTotal.begin(), s_measures.loadInputTotal.end()) << ", " \
                                  << *std::max_element(s_measures.loadInputTotal.begin(), s_measures.loadInputTotal.end()) << ") us, once: (" \
                                  << timeLoadInput << ", " \
                                  << *std::min_element(s_measures.loadInput.begin(), s_measures.loadInput.end()) << ", " \
                                  << *std::max_element(s_measures.loadInput.begin(), s_measures.loadInput.end()) << ") us\n";
  std::cout << "  - Detect total: (" << timeDetectTotal << ", " \
                                  << *std::min_element(s_measures.detectTotal.begin(), s_measures.detectTotal.end()) << ", " \
                                  << *std::max_element(s_measures.detectTotal.begin(), s_measures.detectTotal.end()) << ") us, once: (" \
                                  << timeDetect << ", " \
                                  << *std::min_element(s_measures.detect.begin(), s_measures.detect.end()) << ", " \
                                  << *std::max_element(s_measures.detect.begin(), s_measures.detect.end()) << ") us\n";
  std::cout << "  - Get output total: (" << timeGetOutputTotal << ", " \
                                  << *std::min_element(s_measures.getOutputTotal.begin(), s_measures.getOutputTotal.end()) << ", " \
                                  << *std::max_element(s_measures.getOutputTotal.begin(), s_measures.getOutputTotal.end()) << ") us, once: (" \
                                  << timeGetOutput << ", " \
                                  << *std::min_element(s_measures.getOutput.begin(), s_measures.getOutput.end()) << ", " \
                                  << *std::max_element(s_measures.getOutput.begin(), s_measures.getOutput.end()) << ") us\n";

  std::cout<<"-***-Done! \n";
  return 0;
}

