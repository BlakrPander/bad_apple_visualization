#ifndef LIC_H
#define LIC_H

#include <opencv2/opencv.hpp>

class LineIntegralConvolution{
    private:
        int licLength; // LIC核长度
        float traceStreamline(const cv::Mat& flow, const cv::Mat& noise, int startX, int startY, int direction);
        int countSteps(const cv::Mat& flow, int startX, int startY, int direction);

    public:
        LineIntegralConvolution(int length = 20) : licLength(length) {}
        cv::Mat applyLIC(const cv::Mat& flow, const cv::Mat& noiseTexture);
};

#endif