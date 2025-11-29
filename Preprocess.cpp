#include "Preprocess.hpp"

cv::Mat preprocessFrame(const cv::Mat& frame) {
    cv::Mat gray, binary;
    
    // 转换为灰度图
    if (frame.channels() > 1) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }
    
    // 可选：高斯模糊减少噪声
    // cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    
    // 二值化 - Bad Apple本身就是高对比度，阈值可以设高一些
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);
    
    return binary;
}