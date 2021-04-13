#pragma once

#include "absl/types/optional.h"
#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/reduction.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class BinaryCrossEntropyBackward : public Node {
 public:
  BinaryCrossEntropyBackward(const Value& grad_output, const Value& logits,
                             const Value& labels,
                             const absl::optional<Value>& weight,
                             ReductionMode reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
