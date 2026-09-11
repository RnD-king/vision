// vision/src/yolo_trt_engine.cpp
//
// Pipeline step 2 and 4.
// line_perception_node가 호출하는 TensorRT .engine 로드/추론 담당.
// step 2: BGR image를 yolo_preprocess.cu로 전처리하고 TensorRT enqueue 실행.
// step 4: output [x1,y1,x2,y2,conf,class]를 원본 이미지 좌표 Detection으로
// 복원.
//
// Ultralytics YOLO26 end-to-end 엔진 기준:
// - engine 파일 앞의 Ultralytics metadata header를 건너뜀
// - 입력:  (1, 3, 640, 640) FLOAT
// - 출력:  (1, N, 6) FLOAT = [x1, y1, x2, y2, confidence, class_id]
// - 출력은 이미 end-to-end 결과이므로 C++ NMS를 따로 적용하지 않음

#include "vision/yolo_trt_engine.hpp"
#include "vision/preprocess.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace {

inline void CheckCuda(cudaError_t e, const char *f, int l) {
  if (e != cudaSuccess) {
    std::cerr << "[CUDA] " << cudaGetErrorString(e) << " at " << f << ":" << l
              << std::endl;
    throw std::runtime_error("CUDA failure");
  }
}

#define CHECK_CUDA(x) CheckCuda((x), __FILE__, __LINE__)

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

Logger gLogger;

static void PrintDims(const char *tag, const nvinfer1::Dims &d) {
  std::cout << tag << " nbDims=" << d.nbDims << " [";
  for (int i = 0; i < d.nbDims; ++i) {
    std::cout << d.d[i];
    if (i + 1 < d.nbDims) {
      std::cout << " x ";
    }
  }
  std::cout << "]\n";
}

static bool LooksLikeUltralyticsMetadata(const std::vector<char> &blob,
                                         size_t *engine_offset) {
  if (blob.size() < 8) {
    return false;
  }

  uint32_t metadata_len = 0;
  std::memcpy(&metadata_len, blob.data(), sizeof(metadata_len));

  constexpr uint32_t kMaxReasonableMetadataBytes = 1U << 20;
  const size_t offset =
      sizeof(metadata_len) + static_cast<size_t>(metadata_len);
  if (metadata_len == 0 || metadata_len > kMaxReasonableMetadataBytes ||
      offset >= blob.size()) {
    return false;
  }
  if (blob[sizeof(metadata_len)] != '{') {
    return false;
  }

  *engine_offset = offset;
  return true;
}

static size_t Volume(const nvinfer1::Dims &dims) {
  size_t elems = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return 0;
    }
    elems *= static_cast<size_t>(dims.d[i]);
  }
  return elems;
}

} // namespace

namespace vision {

YoloTrtEngine::YoloTrtEngine(const std::string &engine_path, int input_w,
                             int input_h, float conf_thres)
    : input_w_(input_w), input_h_(input_h), conf_thres_(conf_thres) {
  if (engine_path.empty()) {
    throw std::runtime_error("engine_path is empty");
  }

  if (!LoadEngine(engine_path)) {
    throw std::runtime_error("Failed to load TensorRT engine");
  }

  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_CUDA(cudaEventCreate(&ready_));
}

YoloTrtEngine::~YoloTrtEngine() {
  if (gpu_bgr_) {
    CHECK_CUDA(cudaFree(gpu_bgr_));
  }
  if (d_input_) {
    CHECK_CUDA(cudaFree(d_input_));
  }
  if (d_out_) {
    CHECK_CUDA(cudaFree(d_out_));
  }
  if (h_out_) {
    CHECK_CUDA(cudaFreeHost(h_out_));
  }
  if (ready_) {
    CHECK_CUDA(cudaEventDestroy(ready_));
  }
  if (stream_) {
    CHECK_CUDA(cudaStreamDestroy(stream_));
  }

  context_.reset();
  engine_.reset();
  runtime_.reset();
}

bool YoloTrtEngine::LoadEngine(const std::string &engine_path) {
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "[ERR] Cannot open engine: " << engine_path << std::endl;
    return false;
  }

  const size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> engine_blob(size);
  file.seekg(0, std::ios::beg);
  file.read(engine_blob.data(), static_cast<std::streamsize>(size));

  size_t engine_offset = 0;
  if (LooksLikeUltralyticsMetadata(engine_blob, &engine_offset)) {
    std::cout << "[INFO] Ultralytics TensorRT metadata header detected: "
              << engine_offset << " bytes skipped" << std::endl;
  }

  runtime_.reset(createInferRuntime(gLogger));
  if (!runtime_) {
    std::cerr << "[ERR] createInferRuntime failed" << std::endl;
    return false;
  }

  const void *engine_data = engine_blob.data() + engine_offset;
  const size_t engine_size = engine_blob.size() - engine_offset;
  engine_.reset(runtime_->deserializeCudaEngine(engine_data, engine_size));
  if (!engine_) {
    std::cerr << "[ERR] deserializeCudaEngine failed" << std::endl;
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[ERR] createExecutionContext failed" << std::endl;
    return false;
  }

  const int n_io = engine_->getNbIOTensors();
  for (int i = 0; i < n_io; ++i) {
    const char *name = engine_->getIOTensorName(i);
    const auto mode = engine_->getTensorIOMode(name);
    if (mode == TensorIOMode::kINPUT) {
      in_name_ = name;
    } else if (mode == TensorIOMode::kOUTPUT) {
      out_name_ = name;
    }
  }

  if (in_name_.empty() || out_name_.empty()) {
    std::cerr << "[ERR] Failed to discover I/O tensor names" << std::endl;
    return false;
  }

  auto in_dims_engine = engine_->getTensorShape(in_name_.c_str());
  auto out_dims_engine = engine_->getTensorShape(out_name_.c_str());
  auto in_dtype = engine_->getTensorDataType(in_name_.c_str());
  auto out_dtype = engine_->getTensorDataType(out_name_.c_str());

  bool is_dynamic = false;
  for (int d = 0; d < in_dims_engine.nbDims; ++d) {
    if (in_dims_engine.d[d] == -1) {
      is_dynamic = true;
    }
  }

  if (is_dynamic) {
    nvinfer1::Dims4 fixed{1, 3, input_h_, input_w_};
    if (!context_->setInputShape(in_name_.c_str(), fixed)) {
      std::cerr << "[ERR] setInputShape failed" << std::endl;
      return false;
    }
  }

  const auto in_dims = context_->getTensorShape(in_name_.c_str());
  const auto out_dims = context_->getTensorShape(out_name_.c_str());

  std::cout << "[INFO] Engine loaded: " << engine_path << "\n";
  std::cout << "       in=" << in_name_ << " out=" << out_name_ << "\n";
  PrintDims("       in_dims(engine)  =", in_dims_engine);
  PrintDims("       out_dims(engine) =", out_dims_engine);
  PrintDims("       in_dims(context) =", in_dims);
  PrintDims("       out_dims(context)=", out_dims);
  std::cout << "       in_dtype=" << static_cast<int>(in_dtype)
            << " out_dtype=" << static_cast<int>(out_dtype) << "\n";

  if (in_dtype != DataType::kFLOAT || out_dtype != DataType::kFLOAT) {
    std::cerr << "[ERR] This wrapper expects FLOAT input/output tensors"
              << std::endl;
    return false;
  }
  if (in_dims.nbDims != 4 || in_dims.d[0] != 1 || in_dims.d[1] != 3 ||
      in_dims.d[2] != input_h_ || in_dims.d[3] != input_w_) {
    std::cerr << "[ERR] Input dims mismatch. expected [1x3x" << input_h_ << "x"
              << input_w_ << "]" << std::endl;
    return false;
  }
  if (out_dims.nbDims != 3 || out_dims.d[0] != 1 || out_dims.d[2] != 6) {
    std::cerr << "[ERR] Expected YOLO26 end-to-end output shape [1xNx6]"
              << std::endl;
    return false;
  }

  const size_t in_elems = Volume(in_dims);
  out_elems_ = Volume(out_dims);
  if (in_elems == 0 || out_elems_ == 0) {
    std::cerr << "[ERR] Invalid tensor shape" << std::endl;
    return false;
  }

  CHECK_CUDA(cudaMalloc(&d_input_, in_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out_, out_elems_ * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out_, out_elems_ * sizeof(float)));

  gpu_bgr_ = nullptr;
  gpu_bgr_size_ = 0;

  std::cout << "       out_elems=" << out_elems_ << std::endl;
  return true;
}

void YoloTrtEngine::Preprocess(const cv::Mat &img, float *gpu_input) {
  cv::Mat src = img;
  if (!src.isContinuous()) {
    src = src.clone();
  }

  const int src_w = src.cols;
  const int src_h = src.rows;

  const float scale = std::min(input_w_ / static_cast<float>(src_w),
                               input_h_ / static_cast<float>(src_h));
  const int new_w = static_cast<int>(src_w * scale);
  const int new_h = static_cast<int>(src_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  const size_t required =
      static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * 3;

  if (!gpu_bgr_ || required > gpu_bgr_size_) {
    if (gpu_bgr_) {
      CHECK_CUDA(cudaFree(gpu_bgr_));
    }
    CHECK_CUDA(cudaMalloc(&gpu_bgr_, required));
    gpu_bgr_size_ = required;
  }

  CHECK_CUDA(cudaMemcpyAsync(gpu_bgr_, src.data, required,
                             cudaMemcpyHostToDevice, stream_));

  PreprocessKernelLauncher(gpu_bgr_, src_w, src_h, gpu_input, input_w_,
                           input_h_, new_w, new_h, pad_x, pad_y, scale, stream_,
                           114);
}

bool YoloTrtEngine::Infer(const cv::Mat &img,
                          std::vector<Detection> &detections) {
  if (img.empty() || img.type() != CV_8UC3) {
    detections.clear();
    return false;
  }

  Preprocess(img, reinterpret_cast<float *>(d_input_));

  if (!context_->setTensorAddress(in_name_.c_str(), d_input_)) {
    std::cerr << "[ERR] setTensorAddress(input) failed" << std::endl;
    return false;
  }
  if (!context_->setTensorAddress(out_name_.c_str(), d_out_)) {
    std::cerr << "[ERR] setTensorAddress(output) failed" << std::endl;
    return false;
  }

  if (!context_->enqueueV3(stream_)) {
    std::cerr << "[ERR] enqueueV3 failed" << std::endl;
    return false;
  }

  CHECK_CUDA(cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ready_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ready_));

  Postprocess(detections, img.cols, img.rows);
  return true;
}

void YoloTrtEngine::Postprocess(std::vector<Detection> &detections, int img_w,
                                int img_h) {
  detections.clear();
  if (out_elems_ % 6 != 0) {
    std::cerr << "[ERR] out_elems unexpected for [N,6]. out_elems="
              << out_elems_ << std::endl;
    return;
  }

  const int num = static_cast<int>(out_elems_ / 6);
  const float scale = std::min(input_w_ / static_cast<float>(img_w),
                               input_h_ / static_cast<float>(img_h));
  const int new_w = static_cast<int>(img_w * scale);
  const int new_h = static_cast<int>(img_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  const float *p = h_out_;
  float max_conf = 0.0f;
  int candidates = 0;

  for (int i = 0; i < num; ++i) {
    float x1 = p[0];
    float y1 = p[1];
    float x2 = p[2];
    float y2 = p[3];
    const float conf = p[4];
    const int cls = static_cast<int>(std::round(p[5]));
    p += 6;

    max_conf = std::max(max_conf, conf);
    if (conf < conf_thres_) {
      continue;
    }
    ++candidates;

    x1 = (x1 - static_cast<float>(pad_x)) / scale;
    y1 = (y1 - static_cast<float>(pad_y)) / scale;
    x2 = (x2 - static_cast<float>(pad_x)) / scale;
    y2 = (y2 - static_cast<float>(pad_y)) / scale;

    const int xi = std::clamp(static_cast<int>(std::round(x1)), 0, img_w - 1);
    const int yi = std::clamp(static_cast<int>(std::round(y1)), 0, img_h - 1);
    const int xj = std::clamp(static_cast<int>(std::round(x2)), 0, img_w - 1);
    const int yj = std::clamp(static_cast<int>(std::round(y2)), 0, img_h - 1);

    const int w = std::max(0, xj - xi);
    const int h = std::max(0, yj - yi);
    if (w <= 1 || h <= 1) {
      continue;
    }

    Detection det;
    det.box = cv::Rect(xi, yi, w, h);
    det.confidence = conf;
    det.class_id = cls;
    detections.push_back(det);
  }

  static bool printed = false;
  if (!printed) {
    printed = true;
    std::cout << "[Postprocess] end2end=[1,N,6]"
              << " conf_thres=" << conf_thres_ << " max_conf=" << max_conf
              << " candidates=" << candidates
              << " detections=" << detections.size() << std::endl;
  }
}

} // namespace vision
