#include <iostream>
#include "Farneback.hpp"
#include "LIC.hpp"

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

int main() {
    // 初始化
    cv::VideoCapture cap("bad_apple.mp4");

    if(!cap.isOpened()){
        cerr<<"Error: Couldn't open the video file!"<<endl;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    OpticalFlowAnalyzer flowAnalyzer;
    LineIntegralConvolution licProcessor(25);
    
    // 创建随机噪声纹理
    cv::Mat noiseTexture(frame_height, frame_width, CV_32FC1);
    cv::randn(noiseTexture, 0.5, 0.2); // 均值为0.5，标准差0.2
    
    cv::VideoWriter output("lic_result.avi", 
                          cv::VideoWriter::fourcc('M','J','P','G'), 
                          30, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                      cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    
    cv::Mat frame;
    int framecount=0;
    while (cap.read(frame)) {
        cout<<framecount++<<endl;
        // 计算光流
        cv::Mat flow = flowAnalyzer.computeFlow(frame);

        cv::Mat enhancedFlow;
        flow.convertTo(enhancedFlow, -1, 5.0);  // 放大5倍


        if (!flow.empty()) {
            cv::Mat color = flowAnalyzer.flowToColor(enhancedFlow);
            cv::imshow("original video", color);
            // 应用LIC
            // cv::Mat licResult = licProcessor.applyLIC(flow, noiseTexture);
            
            // 转换为8位用于显示和保存
            cv::Mat licDisplay;
            // licResult.convertTo(licDisplay, CV_8UC1, 255);
            
            // 显示结果
            // cv::imshow("LIC Visualization", licDisplay);
            output.write(licDisplay);
        }
        
        if (cv::waitKey(1) == 27) break; // ESC退出
    }
    
    cap.release();
    output.release();
    return 0;
}