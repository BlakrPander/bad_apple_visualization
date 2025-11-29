#include "LIC.hpp"

//    LineIntegralConvolution(int length = 20) : licLength(length) {}
    
cv::Mat LineIntegralConvolution::applyLIC(const cv::Mat& flow, const cv::Mat& noiseTexture) {
    // 放大光流向量以获得更明显的效果
    cv::Mat enhancedFlow=flow;
    // flow.convertTo(enhancedFlow, -1, 5.0);  // 放大5倍
    
    cv::Mat licResult(flow.size(), CV_32FC1);
    
    #pragma omp parallel for
    for (int y = 0; y < enhancedFlow.rows; y++) {
        for (int x = 0; x < enhancedFlow.cols; x++) {
            float forwardSum = traceStreamline(enhancedFlow, noiseTexture, x, y, 1);
            float backwardSum = traceStreamline(enhancedFlow, noiseTexture, x, y, -1);
            
            // 计算实际追踪的总步数
            int forwardSteps = countSteps(enhancedFlow, x, y, 1);
            int backwardSteps = countSteps(enhancedFlow, x, y, -1);
            int totalSteps = forwardSteps + backwardSteps;
            
            // 避免除零
            if (totalSteps > 0) {
                licResult.at<float>(y, x) = (forwardSum + backwardSum) / totalSteps;
            } else {
                licResult.at<float>(y, x) = noiseTexture.at<float>(y, x); // 使用原始噪声值
            }
        }
    }
    
    return licResult;
}

int LineIntegralConvolution::countSteps(const cv::Mat& flow, int startX, int startY, int direction) {
    int count = 0;
    float x = startX, y = startY;
    
    for (int i = 0; i < licLength; i++) {
        // 检查边界
        if (x < 0 || x >= flow.cols || y < 0 || y >= flow.rows) break;
        
        count++;
        
        // 获取当前点的流向量
        const cv::Vec2f& vec = flow.at<cv::Vec2f>(cvRound(y), cvRound(x));
        
        // 如果向量太小，使用最小步长
        float stepX = direction * vec[0];
        float stepY = direction * vec[1];
        
        // 确保最小移动步长
        if (fabs(stepX) < 0.1) stepX = (stepX >= 0) ? 0.1 : -0.1;
        if (fabs(stepY) < 0.1) stepY = (stepY >= 0) ? 0.1 : -0.1;
        
        // 沿着流线移动
        x += stepX;
        y += stepY;
    }
    
    return count;
}

float LineIntegralConvolution::traceStreamline(const cv::Mat& flow, const cv::Mat& noise, 
                        int startX, int startY, int direction) {
    float sum = 0;
    float x = startX, y = startY;
    
    for (int i = 0; i < licLength; i++) {
        // 检查边界
        if (x < 0 || x >= flow.cols || y < 0 || y >= flow.rows) break;
        
        // 双线性插值获取噪声值
        int x1 = floor(x), y1 = floor(y);
        int x2 = x1 + 1, y2 = y1 + 1;
        float dx = x - x1, dy = y - y1;
        
        // 边界检查
        x1 = std::max(0, std::min(flow.cols-1, x1));
        x2 = std::max(0, std::min(flow.cols-1, x2));
        y1 = std::max(0, std::min(flow.rows-1, y1));
        y2 = std::max(0, std::min(flow.rows-1, y2));
        
        // 双线性插值
        float noiseVal = (1-dx)*(1-dy)*noise.at<float>(y1, x1) +
                        dx*(1-dy)*noise.at<float>(y1, x2) +
                        (1-dx)*dy*noise.at<float>(y2, x1) +
                        dx*dy*noise.at<float>(y2, x2);
        
        sum += noiseVal;
        
        // 获取当前点的流向量
        const cv::Vec2f& vec = flow.at<cv::Vec2f>(y1, x1);
        
        // 如果向量太小，使用最小步长
        float stepX = direction * vec[0];
        float stepY = direction * vec[1];
        
        if (fabs(stepX) < 0.1) stepX = (stepX >= 0) ? 0.1 : -0.1;
        if (fabs(stepY) < 0.1) stepY = (stepY >= 0) ? 0.1 : -0.1;
        
        x += stepX;
        y += stepY;
    }
    
    return sum;  // 返回总和，不在函数内平均
}