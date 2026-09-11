#pragma once

// Pipeline step 3 interface.
// BGR8 camera frame을 YOLO TensorRT 입력 tensor로 변환하는 CUDA launcher.

#include <cstdint>
#include <cuda_runtime.h>

// 전처리 커널 런처
// - 입력: HWC BGR uint8
// - 출력: CHW RGB float32, [0..1] 정규화
// - letterbox: (resized_w, resized_h) 영역을 (pad_x, pad_y) 위치에 배치
// - 패딩 영역은 pad_value(기본 114)로 채움
extern "C" void
PreprocessKernelLauncher(const uint8_t *input_bgr, // [HWC] BGR8 (device)
                         int input_w, int input_h,
                         float *output_chw, // [CHW] float32 (device)
                         int output_w,      // 보통 640
                         int output_h,      // 보통 640
                         int resized_w, int resized_h, int pad_x, int pad_y,
                         float scale, cudaStream_t stream,
                         uint8_t pad_value = 114 // letterbox 패딩 값
);
