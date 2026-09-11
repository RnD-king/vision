#pragma once

// Pipeline step 2/4 interface.
// TensorRT YOLO 엔진을 실행하고 원본 이미지 좌표 Detection 벡터를 반환한다.

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferVersion.h> // ★ 추가: NV_TENSORRT_MAJOR 사용
#include <cuda_runtime.h>

#include "vision/types.hpp"

namespace vision {
// YOLO26 Ultralytics TensorRT 엔진 래퍼.
// 엔진 내부 정밀도는 FP16/FP32 모두 가능하지만, 현재 래퍼는 FLOAT input/output
// tensor를 기대한다.
class YoloTrtEngine {
public:
  // input_w/h: 엔진 입력 해상도 (보통 640x640)
  // conf_thres: postprocess 단계에서 1차로 거를 최소 conf (너무 높게 두면
  // 검출이 사라짐)
  explicit YoloTrtEngine(const std::string &engine_path, int input_w = 640,
                         int input_h = 640, float conf_thres = 0.60f);

  ~YoloTrtEngine();

  YoloTrtEngine(const YoloTrtEngine &) = delete;
  YoloTrtEngine &operator=(const YoloTrtEngine &) = delete;

  // img: BGR8 cv::Mat
  // detections: 출력 벡터 (함수 내부에서 clear 후 채움)
  // 반환: 성공 여부
  bool Infer(const cv::Mat &img, std::vector<Detection> &detections);

private:
  bool LoadEngine(const std::string &engine_path);

  void Preprocess(const cv::Mat &img, float *gpu_input);

  // h_out_를 파싱해서 detections를 채움(원본 해상도 기준으로 unpad/rescale)
  void Postprocess(std::vector<Detection> &detections, int img_w, int img_h);

private:
  // 설정
  int input_w_{640};
  int input_h_{640};
  float conf_thres_{0.60f};

  // TensorRT 객체
  struct TrtDeleter {
    // TensorRT 10+ : destroy() 멤버가 없고 delete로 해제하는 형태(헤더 기준)
    // TensorRT <= 9 : destroy()로 해제
    template <typename T> void operator()(T *p) const noexcept {
      if (!p)
        return;

#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
      delete p;
#else
      p->destroy();
#endif
    }
  };

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};

  std::string in_name_;
  std::string out_name_;

  // CUDA 리소스
  cudaStream_t stream_{nullptr};
  cudaEvent_t ready_{nullptr};

  void *d_input_{nullptr}; // float*
  void *d_out_{nullptr};   // float*
  float *h_out_{nullptr};  // pinned host

  size_t out_elems_{0};

  // 전처리용 GPU staging (BGR8 raw)
  uint8_t *gpu_bgr_{nullptr};
  size_t gpu_bgr_size_{0};
};

} // namespace vision
