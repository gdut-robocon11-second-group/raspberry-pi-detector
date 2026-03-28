#ifndef SERIAL_CONNECTOR_NODE_HPP
#define SERIAL_CONNECTOR_NODE_HPP

#include <asio.hpp>
#include <functional>
#include <future>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <yolo_interfaces/msg/yolo_msg.hpp>
#include <yolo_interfaces/srv/yolo_service.hpp>

#include "serial_connector/transfer_protocol.hpp"

class AsioExecutor : public rclcpp::Executor {
public:
  AsioExecutor() : rclcpp::Executor() {
    // 这里可以初始化 ASIO 相关的成员变量
  }

  void spin() override {
    // 这里可以实现 ASIO 的事件循环，同时调用 rclcpp::Executor 的 spin_some()
    // 来处理 ROS2 事件
    while (rclcpp::ok()) {
      // 处理 ASIO 事件
      io_context_.poll_one();

      // 处理 ROS2 事件
      this->spin_some();
    }
  }

  asio::io_context &get_io_context() { return io_context_; }

  const asio::io_context &get_io_context() const { return io_context_; }

private:
  asio::io_context io_context_;
};

class SerialConnectorNode : public rclcpp::Node {
public:
  SerialConnectorNode() : Node{"serial_connector"} {
    yolo_client_ =
        this->create_client<yolo_interfaces::srv::YOLOService>("yolo_service");
  }

  template <asio::completion_token_for<void(
      std::error_code, yolo_interfaces::srv::YOLOService::Response::SharedPtr)>
                CompletionToken>
  auto request_yolo_detections(CompletionToken &&token) {
    using ResponsePtr = yolo_interfaces::srv::YOLOService::Response::SharedPtr;

    auto request =
        std::make_shared<yolo_interfaces::srv::YOLOService::Request>();

    return asio::async_initiate<CompletionToken,
                                void(std::error_code, ResponsePtr)>(
        [this, request = std::move(request)](auto &&handler) mutable {
          auto h = std::forward<decltype(handler)>(handler);

          if (!yolo_client_ || !yolo_client_->service_is_ready()) {
            auto ex = asio::get_associated_executor(h);
            asio::dispatch(ex, [h = std::move(h)]() mutable {
                h(std::make_error_code(std::errc::not_connected), nullptr);
            });
            return;
          }

          auto shared_handler = std::make_shared<std::decay_t<decltype(h)>>(std::move(h));
          std::function<void(std::shared_future<ResponsePtr>)> callback =
            [shared_handler](std::shared_future<ResponsePtr> future) {
              try {
                (*shared_handler)(std::error_code{}, future.get());
              } catch (...) {
                (*shared_handler)(std::make_error_code(std::errc::io_error), nullptr);
              }
              };

          yolo_client_->async_send_request(request, std::move(callback));
        },
        std::forward<CompletionToken>(token));
  }

private:
  rclcpp::Client<yolo_interfaces::srv::YOLOService>::SharedPtr yolo_client_;
};

#endif // SERIAL_CONNECTOR_NODE_HPP
