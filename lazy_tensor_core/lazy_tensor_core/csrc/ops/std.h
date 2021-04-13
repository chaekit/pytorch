#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Std : public Node {
 public:
  Std(const Value& input, std::vector<lazy_tensors::int64> dimensions,
      bool keep_reduced_dimensions, bool unbiased);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& dimensions() const {
    return dimensions_;
  }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  bool unbiased() const { return unbiased_; }

 private:
  std::vector<lazy_tensors::int64> dimensions_;
  bool keep_reduced_dimensions_;
  bool unbiased_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
