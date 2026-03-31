#ifndef YOLO_NODE_HPP
#define YOLO_NODE_HPP

#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <vector>


#include "camera.hpp"
#include "yolo_interfaces/msg/yolo_msg.hpp"
#include "yolo_interfaces/srv/qr_code_service.hpp"
#include "yolo_interfaces/srv/yolo_service.hpp"
#include "yolo_model.hpp"


class YoloService : public rclcpp::Node {
public:
  YoloService(std::shared_ptr<CameraProxy> camera)
      : Node{"yolo_service"}, camera_(camera) {
    if (!camera_) {
      RCLCPP_ERROR(this->get_logger(), "CameraProxy is null");
      throw std::runtime_error("CameraProxy is null");
    }

    // 从 ROS2 参数读取模型路径，可由 YAML 覆盖
    this->declare_parameter<std::string>("model_path",
                                         "models/best_block_int8.onnx");
    const std::string model_path =
        this->get_parameter("model_path").as_string();

    // 检查模型文件是否存在
    if (!std::filesystem::exists(model_path)) {
      RCLCPP_ERROR(this->get_logger(), "Model file not found: %s",
                   model_path.c_str());
      throw std::runtime_error("Model file not found: " + model_path);
    }

    yolo_detector_ = std::make_unique<YoloOnnxDetector>(model_path);

    RCLCPP_INFO(this->get_logger(), "Using model: %s", model_path.c_str());

    yolo_service_ = this->create_service<yolo_interfaces::srv::YOLOService>(
        "yolo_service",
        std::bind(&YoloService::yolo_callback, this, std::placeholders::_1,
                  std::placeholders::_2));
    // 同时创建 QR code 服务
    qr_code_service_ =
        this->create_service<yolo_interfaces::srv::QRCodeService>(
            "qrcode_service",
            std::bind(&YoloService::qrcode_callback, this,
                      std::placeholders::_1, std::placeholders::_2));
  }

  ~YoloService() = default;

protected:
  void yolo_callback(
      const std::shared_ptr<yolo_interfaces::srv::YOLOService::Request> request,
      std::shared_ptr<yolo_interfaces::srv::YOLOService::Response> response) {
    (void)request;
    RCLCPP_INFO(this->get_logger(), "Received YOLO request");
    // 这里可以调用 YoloOnnxDetector 来处理图像并填充 response->detections
    cv::Mat frame{camera_->getFrame()};
    if (frame.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
      response->detections.clear();
      return;
    }

    const std::vector<Detection> detections = yolo_detector_->detect(frame);
    for (const auto &det : detections) {
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

  void qrcode_callback(
      const std::shared_ptr<yolo_interfaces::srv::QRCodeService::Request>
          request,
      std::shared_ptr<yolo_interfaces::srv::QRCodeService::Response> response) {
    (void)request;
    RCLCPP_INFO(this->get_logger(), "Received QR code request");
    // 这里可以调用 QR code 检测逻辑来处理图像并填充 response->data

    cv::Mat frame{camera_->getFrame()};
    if (frame.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
      response->qr_code_data.clear();
      return;
    }

    try {
      response->qr_code_data = qr_decoder_.detectAndDecode(frame);
      if (response->qr_code_data.empty()) {
        RCLCPP_INFO(this->get_logger(), "No QR code detected");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Decoded QR code data: %s",
                  response->qr_code_data.c_str());
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to decode QR code: %s",
                   e.what());
      response->qr_code_data.clear();
    }
  }

private:
  rclcpp::Service<yolo_interfaces::srv::YOLOService>::SharedPtr yolo_service_;
  rclcpp::Service<yolo_interfaces::srv::QRCodeService>::SharedPtr
      qr_code_service_;
  std::shared_ptr<CameraProxy> camera_;
  std::unique_ptr<YoloOnnxDetector> yolo_detector_;
  cv::QRCodeDetector qr_decoder_;
};

#endif // YOLO_NODE_HPP
