#ifndef YOLO_MODEL_HPP
#define YOLO_MODEL_HPP

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// letterbox 结果：
// image: 缩放+填充后的图
// ratio: 缩放比例（原图 -> 网络输入图）
// dw/dh: 左右与上下方向的填充偏移（用于把框映射回原图）
struct LetterboxResult {
    cv::Mat image;
    float ratio;
    float dw;
    float dh;
};

// 单个检测框的统一表示
struct Detection {
    int classId = -1;
    std::string className;
    float score = 0.0f;
    cv::Rect box;
};

// 封装 ONNXRuntime + YOLO 前后处理
class YoloOnnxDetector {
public:
    // 构造函数：加载模型、读取输入输出信息
    explicit YoloOnnxDetector(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolo-onnx-cpp"),
          sessionOptions_(),
          session_(nullptr),
          memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        // 打开图优化，可减少推理耗时
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // 线程数可按设备调整；嵌入式常见做法是先设 1，再按性能调参
        sessionOptions_.SetIntraOpNumThreads(1);

        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNameAllocated = session_.GetInputNameAllocated(0, allocator);
        auto outputNameAllocated = session_.GetOutputNameAllocated(0, allocator);
        inputName_ = inputNameAllocated.get();
        outputName_ = outputNameAllocated.get();

        // 一般是 [1, 3, H, W]
        auto inputInfo = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> inputShape = inputInfo.GetShape();
        inputH_ = (inputShape.size() >= 4 && inputShape[2] > 0) ? static_cast<int>(inputShape[2]) : 320;
        inputW_ = (inputShape.size() >= 4 && inputShape[3] > 0) ? static_cast<int>(inputShape[3]) : 320;

        // 你这个模型是 4 类数字
        classNames_ = {"1", "2", "3", "4"};
    }

    int inputH() const { return inputH_; }
    int inputW() const { return inputW_; }
    const std::string& inputName() const { return inputName_; }

    // 对单帧图像做检测，返回检测框列表
    std::vector<Detection> detect(const cv::Mat& frame, float confThres = 0.25f) {
        // ratio/dw/dh 用于后面坐标反变换
        float ratio = 1.0f;
        float dw = 0.0f;
        float dh = 0.0f;

        // 预处理：letterbox + BGR2RGB + 归一化 + HWC->CHW
        const std::vector<float> inputTensorValues = preprocess(frame, ratio, dw, dh);

        // 构造 ONNX 输入张量 [1,3,H,W]
        std::array<int64_t, 4> tensorShape = {1, 3, inputH_, inputW_};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_,
            const_cast<float*>(inputTensorValues.data()),
            inputTensorValues.size(),
            tensorShape.data(),
            tensorShape.size());

        std::array<const char*, 1> inputNames{inputName_.c_str()};
        std::array<const char*, 1> outputNames{outputName_.c_str()};

        // 执行推理
        std::vector<Ort::Value> outputs = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            1,
            outputNames.data(),
            1);

        // 安全检查：输出必须存在且是 Tensor
        if (outputs.empty() || !outputs[0].IsTensor()) {
            return {};
        }

        auto outInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> outShape = outInfo.GetShape();

        // 这里按 end2end 导出解析：期望 [1, N, 6]
        // 每个候选框为 [x1, y1, x2, y2, score, class_id]
        if (outShape.size() != 3 || outShape[0] != 1 || outShape[2] < 6) {
            return {};
        }

        const float* outData = outputs[0].GetTensorData<float>();
        const size_t numDet = static_cast<size_t>(outShape[1]);
        const size_t detLen = static_cast<size_t>(outShape[2]);

        std::vector<Detection> dets;
        dets.reserve(numDet);

        for (size_t i = 0; i < numDet; ++i) {
            const float score = outData[i * detLen + 4];
            // 低置信度直接过滤
            if (score < confThres) {
                continue;
            }

            // class_id 通常是浮点数，四舍五入后转整型
            int cls = static_cast<int>(std::round(outData[i * detLen + 5]));
            cls = std::max(0, std::min(cls, static_cast<int>(classNames_.size()) - 1));

            float x1 = outData[i * detLen + 0];
            float y1 = outData[i * detLen + 1];
            float x2 = outData[i * detLen + 2];
            float y2 = outData[i * detLen + 3];

            // 从 letterbox 空间映射回原图空间
            x1 = (x1 - dw) / ratio;
            y1 = (y1 - dh) / ratio;
            x2 = (x2 - dw) / ratio;
            y2 = (y2 - dh) / ratio;

            // 防止越界
            int ix1 = std::max(0, std::min(static_cast<int>(std::round(x1)), frame.cols - 1));
            int iy1 = std::max(0, std::min(static_cast<int>(std::round(y1)), frame.rows - 1));
            int ix2 = std::max(0, std::min(static_cast<int>(std::round(x2)), frame.cols - 1));
            int iy2 = std::max(0, std::min(static_cast<int>(std::round(y2)), frame.rows - 1));

            // 非法框过滤
            if (ix2 <= ix1 || iy2 <= iy1) {
                continue;
            }

            Detection d;
            d.classId = cls;
            d.className = classNames_[cls];
            d.score = score;
            d.box = cv::Rect(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
            dets.push_back(std::move(d));
        }

        return dets;
    }

    // 可视化函数：只负责画框和标签
    static void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            const std::string label = det.className + ": " + cv::format("%.2f", det.score);
            cv::putText(
                image,
                label,
                cv::Point(det.box.x, std::max(0, det.box.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2);
        }
    }

private:
    // letterbox：保持长宽比缩放后，再用灰边填充到固定尺寸
    // 这样能减少几何变形，通常比直接拉伸更稳
    static LetterboxResult letterbox(const cv::Mat& src, int newH, int newW, const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
        const int h = src.rows;
        const int w = src.cols;
        const float r = std::min(static_cast<float>(newH) / static_cast<float>(h),
                                 static_cast<float>(newW) / static_cast<float>(w));

        const int nw = static_cast<int>(std::round(w * r));
        const int nh = static_cast<int>(std::round(h * r));

        const float dw = (newW - nw) / 2.0f;
        const float dh = (newH - nh) / 2.0f;

        cv::Mat resized;
        if (w != nw || h != nh) {
            cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = src.clone();
        }

        const int top = static_cast<int>(std::round(dh - 0.1f));
        const int bottom = static_cast<int>(std::round(dh + 0.1f));
        const int left = static_cast<int>(std::round(dw - 0.1f));
        const int right = static_cast<int>(std::round(dw + 0.1f));

        cv::Mat out;
        cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        return {out, r, dw, dh};
    }

    // 预处理细节：
    // 1) letterbox 到模型输入尺寸
    // 2) BGR -> RGB
    // 3) uint8 -> float32，并归一化到 [0,1]
    // 4) HWC -> CHW
    std::vector<float> preprocess(const cv::Mat& bgr, float& ratio, float& dw, float& dh) const {
        const LetterboxResult lb = letterbox(bgr, inputH_, inputW_);
        ratio = lb.ratio;
        dw = lb.dw;
        dh = lb.dh;

        cv::Mat rgb;
        cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);

        cv::Mat f32;
        rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<cv::Mat> channels(3);
        cv::split(f32, channels);

        std::vector<float> inputTensorValues(1ULL * 3ULL * static_cast<size_t>(inputH_) * static_cast<size_t>(inputW_));
        const size_t channelSize = static_cast<size_t>(inputH_) * static_cast<size_t>(inputW_);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(inputTensorValues.data() + c * channelSize, channels[c].data, channelSize * sizeof(float));
        }

        return inputTensorValues;
    }

private:
    // ONNX Runtime 关键对象
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_;
    Ort::MemoryInfo memoryInfo_;

    // 输入输出信息
    std::string inputName_;
    std::string outputName_;
    int inputH_ = 320;
    int inputW_ = 320;

    // 类别名称映射（class_id -> 可读标签）
    std::vector<std::string> classNames_;
};

#endif // YOLO_MODEL_HPP
