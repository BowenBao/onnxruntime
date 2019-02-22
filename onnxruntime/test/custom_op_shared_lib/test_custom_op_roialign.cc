// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/custom_ops_author.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

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

    return Status::OK();
  }
 protected:
  int64_t pooled_height_, pooled_width_, sampling_ratio_;
  float spatial_scale_;
};

ORT_EXPORT KernelsContainer* GetAllKernels() {
  KernelsContainer* kc = new KernelsContainer;

  KernelDefBuilder def_builder;
  def_builder.SetName("RoiAlign")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(9)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) -> OpKernel* { return new ROIAlignKernel(info); };
  KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);
  kc->kernels_list.push_back(std::move(create_info));
  return kc;
}

ORT_EXPORT SchemasContainer* GetAllSchemas() {
  SchemasContainer* sc = new SchemasContainer;
  sc->domain = onnxruntime::kOnnxDomain;
  sc->baseline_opset_version = 7;
  sc->opset_version = 9;

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

  sc->schemas_list.push_back(schema);
  return sc;
}

ORT_EXPORT void FreeKernelsConstainer(KernelsContainer *kc) {
  delete kc;
}

ORT_EXPORT void FreeSchemasContainer(SchemasContainer* sc) {
  delete sc;
}