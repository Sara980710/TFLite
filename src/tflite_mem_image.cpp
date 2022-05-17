#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc != 3) {
    throw std::invalid_argument("Required arguments: \n            "
                                  "-path to image input\n            "
                                  "-convert to float [0,1]");
  }

  const char *imageFileName = argv[1];

  bool convert = false;
  if (std::stoi(argv[2]) == 1) {
    convert = true;
  }

  std::cout<<"Loading image... \n";

  cv::Mat image = cv::imread(imageFileName);
  if (image.empty())
  {
      throw std::runtime_error("Failed to load image");
      exit(-1);
  }
  
  if (convert) {
      image.convertTo(image, CV_32F, 1.0 / 255, 0);
  }

  sleep(20);
  

  std::cout<<"-***-Done! \n";
  return 0;
}