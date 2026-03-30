#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>


class CameraProxy {
public:
  CameraProxy(int camera_index) : cap_(camera_index) {
    if (!cap_.isOpened()) {
      throw std::runtime_error("Could not open camera");
    }
  }

  ~CameraProxy() { release(); }

  cv::Mat getFrame() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!cap_.isOpened()) {
      throw std::runtime_error("Camera is not opened");
    }
    cv::Mat frame;
    cap_ >> frame;
    return frame;
  }

  void release() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!cap_.isOpened()) {
      return;
    }
    cap_.release();
  }

private:
  cv::VideoCapture cap_;
  mutable std::mutex mtx_;
};
