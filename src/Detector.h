#ifndef DETECTOR_H
#define DETECTOR_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <opencv2/opencv.hpp>

class Detector
{
  std::unique_ptr< tflite::FlatBufferModel > model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
 public:
  std::unique_ptr<tflite::Interpreter> interpreter;

  cv::Mat image;
  Detector(const char *model_path, bool gpu, int threads, bool verbose);
  ~Detector();
  void load_image(const char *image_path, int desiredPrecision, bool normalize, bool verbose);
  void detect(bool verbose);
};

#endif