#include <asio.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "serial_connector/serial_connector_node.hpp"
#include "serial_connector/serial_processor.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto executor = std::make_shared<AsioExecutor>();
  auto node = std::make_shared<SerialConnectorNode>();
  executor->add_node(node);
  asio::co_spawn(executor->get_io_context(),
                 SerialProcessor::read_loop("/dev/ttyS0", node),
                 asio::detached);
  executor->spin();
  rclcpp::shutdown();
  return 0;
}
