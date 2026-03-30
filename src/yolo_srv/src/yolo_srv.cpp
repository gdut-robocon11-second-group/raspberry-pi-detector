#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <yolo_srv/camera.hpp>
#include <yolo_srv/yolo_model.hpp>
#include <yolo_srv/yolo_node.hpp>


int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto camera = std::make_shared<CameraProxy>(0);
  auto node = std::make_shared<YoloService>(camera);
  RCLCPP_INFO(node->get_logger(), "YOLO service started");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
