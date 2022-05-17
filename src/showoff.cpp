#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    if (argc != 4) {
        throw std::invalid_argument("Required arguments: \n            "
                                    "-path to TFLite model file \n            "
                                    "-path to image input\n            "
                                    "-method (1 or 2)");
    }
    const char *modelFileName = argv[1];
    const char *imageFileName = argv[2];
    const int method = std::stoi(argv[3]);

    std::unique_ptr< tflite::FlatBufferModel > model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr) {
        throw std::runtime_error("The model was not able to build to tflite::FlatBufferModel");
    }
    
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (interpreter == nullptr){
        throw std::runtime_error("Failed to initiate the interpreter");
    }

    // Enable use of the GPU delegate, remove below lines to get cpu
    // After TFlite >=2.6, initiate the gpu options
    TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();

    TfLiteDelegate* delegate = TfLiteGpuDelegateV2Create(&gpu_options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        throw std::runtime_error("Failed to create GPU deligate");
        exit(-1);
    }

    // Allocating tensors
    if (interpreter->AllocateTensors() != kTfLiteOk){
        throw std::runtime_error("Failed to allocate tensor");
    }
    

    // Read image
    cv::Mat image = cv::imread(imageFileName);
    if (image.empty())
    {
        throw std::runtime_error("Failed to load image");
        exit(-1);
    }
    
    int input = interpreter->inputs()[0];
    int* inDims = interpreter->tensor(input)->dims->data;
    std::vector<cv::Rect> tiles;
    int currentTile;

    // Calculate how to tile image
    for (int x=0; x < image.size().width; x += inDims[2]) {
        for (int y=0; y < image.size().height; y += inDims[1]) {
        // Handle edges
        if (x + inDims[2] >= image.size().width && y + inDims[1] >= image.size().height) {
            tiles.push_back(cv::Rect(image.size().width - inDims[2], image.size().height - inDims[1], inDims[2], inDims[1]));
        } else {
            if (x + inDims[2] >= image.size().width) {
            tiles.push_back(cv::Rect(image.size().width - inDims[2], y, inDims[2], inDims[1]));
            } else if (y + inDims[1] >= image.size().height) {
            tiles.push_back(cv::Rect(x, image.size().height - inDims[1], inDims[2], inDims[1]));
            } else {
            tiles.push_back(cv::Rect(x, y, inDims[2], inDims[1]));
            }
        }
        }
    }

    //Invoke
    while (currentTile != -1) {
        if (method == 1) {
            int tileByteSize = inDims[1] * inDims[2] * inDims[3] * 4;

            for (int i=0; i < inDims[0]; i++) {
                cv::Mat tile = image(tiles[currentTile]);
                tile.convertTo(tile, CV_32F, 1.0 / 255, 0);
                memcpy(interpreter->typed_tensor<float>(0)+tileByteSize*i, tile.data, tileByteSize); 
            }
        } else if (method == 2) {
            for (int i=0; i < inDims[0]; i++) {
                cv::Mat tile = image(tiles[currentTile]);
                float* input_tensor_float = interpreter->typed_tensor<float>(0);
            
                for (int idx = 0; idx < tile.size[1] * tile.size[0]; idx++){
                    int col = idx % tile.size[0];
                    int row = idx / tile.size[0];
                    cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);
                    
                    input_tensor_float[(inDims[1] * inDims[2] * inDims[3]*i+idx*3+0)] = float(intensity.val[0])/ 255; //R <- B
                    input_tensor_float[(inDims[1] * inDims[2] * inDims[3]*i+idx*3+1)] = float(intensity.val[1])/ 255; //G <- G
                    input_tensor_float[(inDims[1] * inDims[2] * inDims[3]*i+idx*3+2)] = float(intensity.val[2])/ 255; //B <- R
                }
            }
        }

        currentTile = inDims[0] + currentTile;
        if (currentTile != -1 && currentTile >= tiles.size()) {
            currentTile = -1;
        }
        m_interpreter->Invoke();
    }
}
