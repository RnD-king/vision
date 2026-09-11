// vision/src/yolo_preprocess.cu
//
// Pipeline step 3.
// yolo_trt_engine.cpp 내부 Preprocess()에서 호출되는 CUDA 전처리 커널.
// 카메라 BGR8 이미지를 YOLO 입력 형식인 RGB CHW float [0,1]로 변환하고,
// letterbox resize/padding을 적용한다.

#include "vision/preprocess.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// output(640x640) 전체를 돌면서,
// (pad_x..pad_x+resized_w-1, pad_y..pad_y+resized_h-1)만 입력에서 샘플링하고
// 나머지는 pad_value로 채운다.
__global__ void
PreprocessKernel(const uint8_t *__restrict__ input_bgr, // device, HWC BGR
                 int input_w, int input_h,
                 float *__restrict__ output_chw, // device, CHW RGB float
                 int output_w, int output_h, int resized_w, int resized_h,
                 int pad_x, int pad_y,
                 float inv_scale, // 1/scale
                 float pad_f      // pad_value/255.0f
) {
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= output_w || y >= output_h)
    return;

  const int out_idx = y * output_w + x;
  const int area = output_w * output_h;

  // 기본: 패딩 값으로 채움
  float r = pad_f, g = pad_f, b = pad_f;

  // resized 영역 내부면 입력에서 샘플링
  const int rx = x - pad_x;
  const int ry = y - pad_y;
  if (0 <= rx && rx < resized_w && 0 <= ry && ry < resized_h) {
    // letterbox 역변환: resized 좌표 -> 원본 좌표
    // nearest neighbor
    const float src_x_f = (rx + 0.5f) * inv_scale - 0.5f;
    const float src_y_f = (ry + 0.5f) * inv_scale - 0.5f;

    int sx = (int)lrintf(src_x_f);
    int sy = (int)lrintf(src_y_f);

    sx = clamp_int(sx, 0, input_w - 1);
    sy = clamp_int(sy, 0, input_h - 1);

    const int in_idx = (sy * input_w + sx) * 3;
    const uint8_t bb = input_bgr[in_idx + 0];
    const uint8_t gg = input_bgr[in_idx + 1];
    const uint8_t rr = input_bgr[in_idx + 2];

    // BGR -> RGB, [0..1]
    r = (float)rr * (1.0f / 255.0f);
    g = (float)gg * (1.0f / 255.0f);
    b = (float)bb * (1.0f / 255.0f);
  }

  // CHW (R, G, B)
  output_chw[out_idx] = r;
  output_chw[out_idx + area] = g;
  output_chw[out_idx + area * 2] = b;
}

} // namespace

extern "C" void
PreprocessKernelLauncher(const uint8_t *input_bgr, int input_w, int input_h,
                         float *output_chw, int output_w, int output_h,
                         int resized_w, int resized_h, int pad_x, int pad_y,
                         float scale, cudaStream_t stream, uint8_t pad_value) {
  const dim3 threads(16, 16);
  const dim3 blocks((output_w + threads.x - 1) / threads.x,
                    (output_h + threads.y - 1) / threads.y);

  const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 1.0f;
  const float pad_f = (float)pad_value * (1.0f / 255.0f);

  PreprocessKernel<<<blocks, threads, 0, stream>>>(
      input_bgr, input_w, input_h, output_chw, output_w, output_h, resized_w,
      resized_h, pad_x, pad_y, inv_scale, pad_f);
}
