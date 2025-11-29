#ifndef FARNEBACK_H
#define FARNEBACK_H

#include <opencv2/opencv.hpp>
#include "Preprocess.hpp"  // 包含头文件，不是.cpp文件

class OpticalFlowAnalyzer {
private:
    cv::Mat prevGray;
    cv::Mat flow;
    bool firstFrame = true;

public:
    cv::Mat computeFlow(const cv::Mat& currentFrame);
    cv::Mat flowToColor(const cv::Mat& flow);
    cv::Mat interpolateFlow(const cv::Mat& flow, const cv::Mat& currentFrame);
};

#endif // FARNEBACK_H