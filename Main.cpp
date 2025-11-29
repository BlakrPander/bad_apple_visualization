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
        return -1;
    }

    // 获取原始视频参数
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    cout << "Video info: " << frame_width << "x" << frame_height 
         << ", FPS: " << fps << endl;

    OpticalFlowAnalyzer flowAnalyzer;
    LineIntegralConvolution licProcessor(25);
    
    // 创建随机噪声纹理
    cv::Mat noiseTexture(frame_height, frame_width, CV_32FC1);
    cv::randn(noiseTexture, 0.5, 0.2);
    
    // 创建VideoWriter - 使用灰度输出
    cv::VideoWriter output("lic_result.avi", 
                          cv::VideoWriter::fourcc('M','J','P','G'), 
                          fps, 
                          cv::Size(frame_width, frame_height),
                          false); // false表示灰度视频
    
    if (!output.isOpened()) {
        cerr << "Error: Could not open video writer!" << endl;
        return -1;
    }
    
    cv::Mat frame;
    int framecount = 0;
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    cout << "Processing " << total_frames << " frames..." << endl;
    
    while (cap.read(frame)) {
        cout << "Processing frame " << ++framecount << "/" << total_frames << endl;
        
        // 计算光流
        cv::Mat rawflow = flowAnalyzer.computeFlow(frame);
        cv::Mat flow = flowAnalyzer.interpolateFlow(rawflow, frame);

        if (!flow.empty()) {
            // 应用LIC
            cv::Mat licResult = licProcessor.applyLIC(flow, noiseTexture);
            
            // 转换为8位并直接保存
            cv::Mat licDisplay;
            licResult.convertTo(licDisplay, CV_8UC1, 255);
            output.write(licDisplay);
        }
        
        // if(framecount == 200)
        //     break;
    }
    
    cap.release();
    output.release();
    
    cout << "Processing completed! Output saved to lic_result.avi" << endl;
    
    return 0;
}