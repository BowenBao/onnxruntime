#pragma once

#include "core/framework/data_types.h"
#include "core/graph/graph.h"
#include "onnx/defs/data_type_utils.h"

namespace onnxruntime {
namespace Utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg);
}  // namespace Utils
}  // namespace onnxruntime