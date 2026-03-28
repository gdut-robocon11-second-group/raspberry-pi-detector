#ifndef SERIAL_PROCESSOR_HPP
#define SERIAL_PROCESSOR_HPP

#include <asio.hpp>
#include <cmath>
#include <numeric>
#include <rclcpp/rclcpp.hpp>
#include <span>
#include <string>
#include <vector>
#include <yolo_interfaces/msg/yolo_msg.hpp>
#include <yolo_interfaces/srv/yolo_service.hpp>

#include "serial_connector/serial_connector_node.hpp"

enum class SerialMessageType : uint16_t {
  YOLO_DETECTION = 0x0001,
  // 可以在这里定义更多的消息类型
};

class SerialProcessor {
public:
  SerialProcessor(asio::any_io_executor &io_context,
                  const std::string &port_name,
                  std::shared_ptr<SerialConnectorNode> node)
      : serial_port_(io_context, port_name), node_(node) {
    serial_port_.set_option(asio::serial_port_base::baud_rate(115200));
    serial_port_.set_option(asio::serial_port_base::character_size(8));
    serial_port_.set_option(
        asio::serial_port_base::parity(asio::serial_port_base::parity::none));
    serial_port_.set_option(asio::serial_port_base::stop_bits(
        asio::serial_port_base::stop_bits::one));
    serial_port_.set_option(asio::serial_port_base::flow_control(
        asio::serial_port_base::flow_control::none));

    if (!serial_port_.is_open()) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to open serial port: %s",
                   port_name.c_str());
      return;
    } else {
      RCLCPP_INFO(node_->get_logger(), "Serial port opened: %s",
                  port_name.c_str());
    }

    packet_mgr_.set_receive_function(
        [this](gdut::data_packet<gdut::crc16_algorithm> packet) {
          // 这里可以处理接收到的完整数据包，例如解析协议、发布 ROS2 消息等
          // 注意：这个回调可能在 ASIO 的线程中被调用，如果需要与 ROS2
          // 交互，可能需要使用 rclcpp::Publisher 或者
          // rclcpp::Node::get_logger() 等线程安全的接口
          asio::co_spawn(serial_port_.get_executor(),
                         process(std::move(packet)), asio::detached);
        });
  }

  bool is_open() const { return serial_port_.is_open(); }

  static asio::awaitable<void>
  read_loop(const std::string &port_name,
            std::shared_ptr<SerialConnectorNode> node) {
    auto executor = co_await asio::this_coro::executor;
    SerialProcessor processor{executor, port_name, node};
    if (!processor.is_open()) {
      RCLCPP_ERROR(node->get_logger(), "Failed to open serial port");
      co_return;
    }
    std::array<uint8_t, 4096> data;
    while (true) {
      std::size_t n = co_await processor.serial_port_.async_read_some(
          asio::buffer(data), asio::use_awaitable);
      // 处理读取到的数据
      co_await processor.received_data(
          std::span<const uint8_t>(data.data(), n));
    }
  }

protected:
  asio::awaitable<void> received_data(std::span<const uint8_t> data) {
    // 这里可以处理接收到的数据，例如解析协议、发布 ROS2 消息等
    packet_mgr_.receive(data.begin(), data.end());
    co_return;
  }

  asio::awaitable<void> process(gdut::data_packet<gdut::crc16_algorithm> packet) {
    // 这里可以处理一个完整的数据包，例如解析协议、发布 ROS2 消息等
    RCLCPP_INFO(node_->get_logger(),
                "Processing a complete packet with code: %u", packet.code());
    if (packet.code() ==
        static_cast<uint16_t>(SerialMessageType::YOLO_DETECTION)) {
      // 请求 YOLO 检测结果
      try {
        auto response =
            co_await node_->request_yolo_detections(asio::use_awaitable);
        if (!response) {
          RCLCPP_ERROR(node_->get_logger(), "Failed to get YOLO detections");
          co_return;
        }
        // 这里可以处理 YOLO 检测结果，例如将其封装成数据包发送回串口，或者发布
        // ROS2 消息等
        double distance = std::numeric_limits<double>::max();
        int16_t class_id = -1;
        for (auto &yolo_msg : response->detections) {
          double obj_distance =
              std::hypot(yolo_msg.box_x + yolo_msg.box_width / 2.0 - 320.0,
                         yolo_msg.box_y + yolo_msg.box_height / 2.0 - 320.0);
          if (obj_distance < distance && yolo_msg.score > 0.25) {
            distance = obj_distance;
            class_id = yolo_msg.class_id;
          }
        }
        if (class_id == -1) {
          RCLCPP_WARN(node_->get_logger(), "No valid YOLO detections found");
          co_return;
        }
        std::string class_name{std::to_string(class_id + 1)};
        gdut::data_packet<gdut::crc16_algorithm> response_packet{
            static_cast<uint16_t>(SerialMessageType::YOLO_DETECTION),
            class_name.begin(), class_name.end(), gdut::build_packet};
        co_await serial_port_.async_write_some(
            asio::buffer(response_packet.data(), response_packet.size()),
            asio::use_awaitable);
      } catch (const std::exception &e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Error occurred while requesting YOLO detections: %s",
                     e.what());
      }
      RCLCPP_INFO(node_->get_logger(), "Received YOLO detection packet");
    } else {
      RCLCPP_WARN(node_->get_logger(), "Received packet with unknown code: %u",
                  packet.code());
    }
    co_return;
  }

private:
  asio::serial_port serial_port_;
  std::shared_ptr<SerialConnectorNode> node_;
  gdut::packet_manager<gdut::crc16_algorithm> packet_mgr_;
};

#endif // SERIAL_PROCESSOR_HPP
