#include <iostream>
#include "Detector.h"
#include <numeric>

struct Measures{
  std::vector<float> detect;
  std::vector<float> detectTotal;
};

static Measures s_measures;
  
float getAverage(std::vector<float> timeMeasures) {
    auto const count = static_cast<float>(timeMeasures.size());
    float duration = std::accumulate(timeMeasures.begin(), timeMeasures.end(), 0) / count;
    return duration;
}

void run_once(const char *modelFileName, bool gpu, int threads, bool verbose, const char *imageFileName, int desiredPrecision, bool normalize, const uint8_t method){
  
  Detector detector(modelFileName, gpu, threads, verbose);
  detector.load_image(imageFileName, desiredPrecision, normalize, verbose, method);
  detector.tile_image(verbose);

  auto t1 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

  // Run
  std::vector<float> t_detect;
  while (detector.currentTile != -1) {
    
    detector.load_input(verbose, method); // Does not load any image values
    t1 = std::chrono::high_resolution_clock::now();
    detector.detect(verbose);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    t_detect.push_back(duration);
  }
  s_measures.detect.push_back(getAverage(t_detect));
  s_measures.detectTotal.push_back(std::accumulate(t_detect.begin(), t_detect.end(), 0));
}

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cout<<argc<<std::endl;
    throw std::invalid_argument("Required arguments: \n            "
                                  "-path to TFLite model file \n            "
                                  "-iterations for stable measures\n            "
                                  "-desired precision (0:f32, 1:f16, 2:int8)\n            "
                                  "-path to image input\n            "
                                  "-Number of threads\n            "
                                  "-device (0:cpu or 1:gpu)\n            "
                                  "-verbose (0:false, 1:true)");
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
  if (std::stoi(argv[7]) == 1) {
    verbose = true;
  }
  
  bool normalize = true;
  const uint8_t method = 3; 

  std::cout<< "Running " << method << "method" <<std::endl;
  std::cout<< "GPU: " << gpu <<std::endl;
  std::cout<< "threads (CPU): " << threads <<std::endl;
  std::cout << "Total iterations: " << iterations << std::endl;
  std::cout << "Iteration progress: " << std::flush;
  for (int i=0; i < iterations; i++) {
    std::cout << i + 1<< ", " << std::flush;
    run_once(modelFileName, gpu, threads, verbose, imageFileName, desiredPrecision, normalize, method);
  }


  float timeDetect = getAverage(s_measures.detect);
  float timeDetectTotal = getAverage(s_measures.detectTotal);

  std::cout << std::fixed;
  std::cout << "\nMeasures detail, (average, min, max): \n";
  std::cout << "  - Detect total: (" << timeDetectTotal << ", " \
                                  << *std::min_element(s_measures.detectTotal.begin(), s_measures.detectTotal.end()) << ", " \
                                  << *std::max_element(s_measures.detectTotal.begin(), s_measures.detectTotal.end()) << ") us, once: (" \
                                  << timeDetect << ", " \
                                  << *std::min_element(s_measures.detect.begin(), s_measures.detect.end()) << ", " \
                                  << *std::max_element(s_measures.detect.begin(), s_measures.detect.end()) << ") us\n";

  std::cout<<"-***-Done! \n"<<std::endl;
  return 0;
}

