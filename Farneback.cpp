#include "Farneback.hpp"

cv::Mat OpticalFlowAnalyzer::computeFlow(const cv::Mat& currentFrame) {
    cv::Mat currentGray = preprocessFrame(currentFrame);
    
    if (firstFrame) {
        prevGray = currentGray.clone();
        firstFrame = false;
        return cv::Mat(); // 第一帧没有光流
    }
    
    // 计算Farneback稠密光流
    cv::calcOpticalFlowFarneback(
        prevGray, currentGray, flow,
        0.5,  // pyramid scale
        3,    // levels
        15,   // winsize
        3,    // iterations
        5,    // poly_n
        1.2,  // poly_sigma
        0     // flags
    );
    
    prevGray = currentGray.clone();
    return flow;
}

// 将光流场转换为可视化的RGB图像（调试用）
cv::Mat OpticalFlowAnalyzer::flowToColor(const cv::Mat& flow) {
    if (flow.empty()) {
        return cv::Mat::zeros(480, 640, CV_8UC3); // 返回默认大小的黑色图像
    }
    
    cv::Mat flowColor;
    cv::Mat flowSplit[2];
    cv::split(flow, flowSplit);
    
    // 计算幅度和角度
    cv::Mat magnitude, angle;
    cv::cartToPolar(flowSplit[0], flowSplit[1], magnitude, angle, true);
    
    // 归一化幅度用于可视化
    double magMax;
    cv::minMaxLoc(magnitude, 0, &magMax);
    if (magMax > 0)
        magnitude.convertTo(magnitude, -1, 1.0 / magMax);
    
    // 创建HSV图像然后转RGB
    cv::Mat hsv(flow.size(), CV_8UC3);
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(
                angle.at<float>(y, x) * 180 / CV_PI / 2,  // H
                255,                                      // S
                cv::saturate_cast<uchar>(magnitude.at<float>(y, x) * 255) // V
            );
        }
    }
    
    cv::cvtColor(hsv, flowColor, cv::COLOR_HSV2BGR);
    return flowColor;
}

cv::Mat OpticalFlowAnalyzer::interpolateFlow(const cv::Mat& flow, const cv::Mat& currentFrame) {
    if(flow.empty())
        return flow;
    cv::Mat flowsplit[2];
    cv::split(flow, flowsplit);
    
    // 创建有效光流掩码
    cv::Mat validFlowMask = cv::Mat::zeros(flow.size(), CV_8UC1);
    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            cv::Vec2f vec = flow.at<cv::Vec2f>(y, x);
            if (cv::norm(vec) > 0.1) {
                validFlowMask.at<uchar>(y, x) = 255;
            }
        }
    }
    
    cv::Mat binaryMask = currentFrame;
    if (currentFrame.channels() > 1) {
    cv::cvtColor(currentFrame, binaryMask, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryMask, binaryMask, 128, 255, cv::THRESH_BINARY);
}

    // 找到物体内部区域（二值图像中为前景但没有有效光流的区域）
    cv::Mat internalRegion = binaryMask & (~validFlowMask);
    
    // 对内部区域进行距离变换，计算到最近边界的距离
    cv::Mat dist;
    cv::distanceTransform(internalRegion, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    
    // 只对内部区域进行插值
    cv::Mat interpolated_flow_x, interpolated_flow_y;
    flowsplit[0].copyTo(interpolated_flow_x);
    flowsplit[1].copyTo(interpolated_flow_y);
    
    // 迭代传播：从边界向内部逐步传播光流
    for (int iter = 0; iter < 10; iter++) {
        cv::Mat temp_x = interpolated_flow_x.clone();
        cv::Mat temp_y = interpolated_flow_y.clone();
        
        for (int y = 1; y < flow.rows - 1; y++) {
            for (int x = 1; x < flow.cols - 1; x++) {
                // 只处理内部区域且当前没有有效光流的点
                if (internalRegion.at<uchar>(y, x) && cv::norm(flow.at<cv::Vec2f>(y, x)) < 0.1) {
                    cv::Vec2f sum(0, 0);
                    int count = 0;
                    
                    // 检查4邻域
                    int dx[] = {0, 1, 0, -1};
                    int dy[] = {1, 0, -1, 0};
                    
                    for (int k = 0; k < 4; k++) {
                        int nx = x + dx[k];
                        int ny = y + dy[k];
                        
                        if (nx >= 0 && nx < flow.cols && ny >= 0 && ny < flow.rows) {
                            cv::Vec2f neighbor = cv::Vec2f(
                                temp_x.at<float>(ny, nx),
                                temp_y.at<float>(ny, nx)
                            );
                            
                            // 如果邻居有有效光流
                            if (cv::norm(neighbor) > 0.1 || 
                                dist.at<float>(ny, nx) < dist.at<float>(y, x)) {
                                sum += neighbor;
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0) {
                        interpolated_flow_x.at<float>(y, x) = sum[0] / count;
                        interpolated_flow_y.at<float>(y, x) = sum[1] / count;
                    }
                }
            }
        }
    }
    
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{interpolated_flow_x, interpolated_flow_y}, result);
    return result;
}