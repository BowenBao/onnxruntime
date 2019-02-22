// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include "core/graph/schema_registry.h"
#include "core/framework/customregistry.h"
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

class ROIAlignKernel : public OpKernel {
 public:
  ROIAlignKernel(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<int64_t> pooled_shape;
    ORT_ENFORCE(info.GetAttrs<int64_t>("pooled_shape", pooled_shape).IsOK());
    ORT_ENFORCE(pooled_shape.size() == 2);

    pooled_height_ = pooled_shape[0];
    pooled_width_ = pooled_shape[1];

    ORT_ENFORCE(info.GetAttr<float>("spatial_scale", &spatial_scale_).IsOK());

    ORT_ENFORCE(info.GetAttr<int64_t>("sampling_ratio", &sampling_ratio_).IsOK());
  }

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* x = ctx->Input<Tensor>(0);
    const Tensor* rois = ctx->Input<Tensor>(1);
    if (x == nullptr || rois == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    auto* x_data = x->template Data<float>();
    auto* rois_data = rois->template Data<float>();

    // TODO: add implementation here.

    (void(x_data));
    (void(rois_data));

    int channels = static_cast<int>(x->Shape()[1]);
    int num_rois = static_cast<int>(rois->Shape()[0]);

    std::vector<int64_t> output_dims({num_rois, channels, pooled_height_, pooled_width_});

    auto* y = ctx->Output(0, TensorShape(output_dims));
    auto* y_data = y->MutableData<float>();

    for (size_t n = 0; n < num_rois * channels * pooled_height_ * pooled_width_; n++) {
      y_data[n] = 0.0f;
    }

    return Status::OK();
  }
 protected:
  int64_t pooled_height_, pooled_width_, sampling_ratio_;
  float spatial_scale_;
};

ONNX_NAMESPACE::OpSchema GetROIAlignSchema() {
  ONNX_NAMESPACE::OpSchema schema("RoiAlign", "unknown", 0);
  schema.Attr(
    "pooled_shape",
    "Description",
    AttributeProto::INTS
  );
  schema.Attr(
    "spatial_scale",
    "Description",
    AttributeProto::FLOAT,
    1.f
  );
  schema.Attr(
    "sampling_ratio",
    "Description",
    AttributeProto::INT,
    static_cast<int64_t>(1)
  );
  schema.Input(0,
               "x",
               "Description",
               "T");
  schema.Input(1,
               "rois",
               "Description",
               "T");
  schema.Output(0, "y", "Description", "T");
  schema.TypeConstraint(
    "T",
    OpSchema::numeric_types_for_math_reduction(),
    "Constrain input and output types to high-precision numeric tensors."
  );
  schema.SinceVersion(9);
  return schema;
}

// Register
KernelDefBuilder ROIAlignKernelDef() {
  KernelDefBuilder def;
  def.SetName("RoiAlign")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(9)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

OpKernel* CreateROIAlignKernel(const OpKernelInfo& kernel_info) {
  return new ROIAlignKernel(kernel_info);
}

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                const std::vector<int64_t>& dims_x,
                const std::vector<float>& values_x,
                const std::vector<int64_t>& dims_rois,
                const std::vector<float>& values_rois,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y) {
  MLValue x_ml_value, rois_ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x, &x_ml_value);
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_rois, values_rois, &rois_ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("x", x_ml_value));
  feeds.insert(std::make_pair("rois", rois_ml_value));

  std::vector<std::string> output_names;
  output_names.push_back("y");
  std::vector<MLValue> fetches;

  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;

  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.template Data<float>(), rtensor.template Data<float>() + expected_shape.Size());
  ASSERT_EQ(values_y, found);
}

static const std::string ROIALIGN_MODEL_URI = "roialign.onnx";

TEST(ROIKernelTests, simple) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  auto roialign_schema = GetROIAlignSchema();
  std::vector<OpSchema> schemas = {roialign_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 7, 9).IsOK());
  auto def = ROIAlignKernelDef();
  EXPECT_TRUE(registry->RegisterCustomKernel(def, CreateROIAlignKernel).IsOK());

  EXPECT_TRUE(session_object.Load(ROIALIGN_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {1, 3, 6, 6};
  std::vector<float> values_x(3*6*6, 1.0f);
  std::vector<int64_t> dims_rois = {3, 5};
  std::vector<float> values_rois(3*5, 1.0f);

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 3, 1, 1};
  std::vector<float> expected_values_y(3*3, 0.0f);

  RunSession(session_object, run_options, dims_x, values_x, dims_rois, values_rois,
             expected_dims_y, expected_values_y);
}

}
}