#ifndef YOLO_NODE_HPP
#define YOLO_NODE_HPP

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>

#include "yolo_srv/yolo_model.hpp"
#include "yolo_interfaces/msg/yolo_msg.hpp"
#include "yolo_interfaces/srv/yolo_service.hpp"

class YoloService : public rclcpp::Node
{
public:
    YoloService() : Node{"yolo_service"}, cap_{0}
    {
        if (!cap_.isOpened()) {
            throw std::runtime_error("Cannot open camera");
        }

        // 从 ROS2 参数读取模型路径，可由 YAML 覆盖
        this->declare_parameter<std::string>("model_path", "models/best_block_int8.onnx");
        const std::string model_path = this->get_parameter("model_path").as_string();

        // 检查模型文件是否存在
        if (!std::filesystem::exists(model_path)) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", model_path.c_str());
            throw std::runtime_error("Model file not found: " + model_path);
        }
        
        detector_ = std::make_unique<YoloOnnxDetector>(model_path);

        RCLCPP_INFO(this->get_logger(), "Using model: %s", model_path.c_str());

        service_ = this->create_service<yolo_interfaces::srv::YOLOService>(
            "yolo_service",
            std::bind(&YoloService::callback, this, std::placeholders::_1, std::placeholders::_2));
    }

    ~YoloService() = default;

protected:
    void callback(const std::shared_ptr<yolo_interfaces::srv::YOLOService::Request> request,
                   std::shared_ptr<yolo_interfaces::srv::YOLOService::Response> response) {
        (void)request;
        (void)response;
        RCLCPP_INFO(this->get_logger(), "Received YOLO request");
        // 这里可以调用 YoloOnnxDetector 来处理图像并填充 response->detections
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
            response->detections.clear();
            return;
        }

        const std::vector<Detection> detections = detector_->detect(frame);
        for (const auto& det : detections) {
            yolo_interfaces::msg::YOLOMsg msg;
            msg.class_id = det.classId;
            msg.class_name = det.className;
            msg.score = det.score;
            msg.box_x = det.box.x;
            msg.box_y = det.box.y;
            msg.box_width = det.box.width;
            msg.box_height = det.box.height;
            response->detections.push_back(msg);
        }
    }

private:
    rclcpp::Service<yolo_interfaces::srv::YOLOService>::SharedPtr service_;
    cv::VideoCapture cap_;
    std::unique_ptr<YoloOnnxDetector> detector_;
};

#endif // YOLO_NODE_HPP
