#ifndef DETECTOR_H
#define DETECTOR_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <opencv2/opencv.hpp>

class Detector
{
 private:
  std::unique_ptr< tflite::FlatBufferModel > m_model;
  tflite::ops::builtin::BuiltinOpResolver m_resolver;
  std::unique_ptr<tflite::Interpreter> m_interpreter;
  std::vector<cv::Rect> m_tiles;
  TfLiteDelegate* m_delegate;

 public:
  cv::Mat image;
  int* inDims;
  int* outDims;
  int currentTile;

  Detector(const char *model_path, bool gpu, int threads, bool verbose);
  ~Detector();

  void load_image(const char *image_path, int desiredPrecision, bool normalize, bool verbose);
  void tile_image(bool verbose);
  void load_input(bool verbose);
  void detect(bool verbose);
  float* get_output(bool verbose);
};

#endif