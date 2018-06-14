﻿#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

#include "gsl/gsl_util"

namespace Lotus {

template <typename T>
class MeanVarianceNormalization final : public OpKernel {
 public:
  MeanVarianceNormalization(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttr<int64_t>("across_channels", &across_channels_).IsOK());
    LOTUS_ENFORCE(info.GetAttr<int64_t>("normalize_variance", &normalize_variance_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const auto dims = X->Shape().GetDims();

    if (dims.size() < 4) {
      return Status(LOTUS, INVALID_ARGUMENT,
                    "Input is expected to have four dimensions corresponding to [N,C,H,W]");
    }

    const int64_t N = dims[0];
    const int64_t C = dims[1];
    const int64_t H = dims[2];
    const int64_t W = dims[3];

    Tensor* Y = context->Output(0, TensorShape({N, C, H, W}));
    const T* Xdata = X->Data<T>();
    T* Ydata = Y->MutableData<T>();

    const int64_t sample_size = H * W;
    Eigen::Array<float, Eigen::Dynamic, 1> mean(C, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> var(C, 1);
    mean.setZero();
    var.setZero();

    ConstEigenArrayMap<T> X_arr(Xdata, sample_size, N * C);
    for (int nc = 0; nc < N * C; ++nc) {
      mean(nc % C) += X_arr.col(nc).sum();
    }
    mean /= gsl::narrow_cast<T>(N * sample_size);
    for (int64_t nc = 0; nc < N * C; ++nc) {
      var(nc % C) += (X_arr.col(nc) - mean(nc % C)).matrix().squaredNorm();
    }
    var /= gsl::narrow_cast<T>(N * sample_size);

    Eigen::Array<T, Eigen::Dynamic, 1> inv_std;
    EigenArrayMap<T> Y_arr(Ydata, sample_size, N * C);

    if (across_channels_) {
      // m_c = sum(m_i) / n
      float global_mean = mean.mean();

      // var_c = [(var_1 + (m_1 - m_c)^2) + ...  + (var_n + (m_n - m_c)^2)] / n
      //       = [sum(var_i) + squared_norm(m_i - m_c)] / n
      float global_var = ((mean - global_mean).matrix().squaredNorm() + var.sum()) / C;

      // For across channels we can directly use eigen because global_mean and global_var
      // are just scalars.
      if (!normalize_variance_) {
        Y_arr = X_arr - global_mean;
      } else {
        float inv_std_scalar = 1 / std::sqrt(global_var);
        Y_arr = (X_arr - global_mean) * inv_std_scalar;
      }
    } else {
      if (!normalize_variance_) {
        // inv_std = 1
        for (int64_t nc = 0; nc < N * C; ++nc) {
          // y = (x - mean)
          Y_arr.col(nc) = (X_arr.col(nc) - mean(nc % C));
        }
      } else {
        inv_std = var.sqrt().inverse();
        for (int64_t nc = 0; nc < N * C; ++nc) {
          // y = (x - mean) * (inv_std)
          Y_arr.col(nc) = (X_arr.col(nc) - mean(nc % C)) * inv_std(nc % C);
        }
      }
    }
    return Status::OK();
  }

 private:
  int64_t across_channels_;
  int64_t normalize_variance_;
};

}  //namespace Lotus
