#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/in_place.h>

#include <type_traits>

namespace c10 {

/// A smart pointer around either a borrowed or owned T. Maintains an
/// internal raw pointer when constructed with borrowed(), with all
/// the attendant lifetime concerns.  Compare to Rust's
/// std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html), but note
/// that it is probably not suitable for general use because C++ has
/// no borrow checking. Included here to support
/// Tensor::expect_contiguous.
template <typename T>
class MaybeOwned final {
  bool isBorrowed_;
  union {
    const T *borrow_;
    T own_;
  };

  /// Don't use this; use borrowed() instead.
  explicit MaybeOwned(const T& t) : isBorrowed_(true), borrow_(&t) {}

  /// Don't use this; use owned() instead.
  explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value)
  : isBorrowed_(false), own_(std::move(t)) {}

  /// Don't use this; use owned() instead.
  template <class... Args>
  explicit MaybeOwned(in_place_t, Args&&... args)
  : isBorrowed_(false)
  , own_(std::forward<Args>(args)...) {}

 public:
  explicit MaybeOwned(): isBorrowed_(true), borrow_(nullptr) {}

  // Copying a borrow yields another borrow of the original, as with a
  // T*. Copying an owned T yields another owned T for safety: no
  // chains of borrowing by default! (Note you could get that behavior
  // with MaybeOwned<T>::borrowed(*rhs) if you wanted it.)
  MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      borrow_ = rhs.borrow_;
    } else {
      new (&own_) T(rhs.own_);
    }
  }

  MaybeOwned& operator=(const MaybeOwned& rhs) {
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        borrow_ = rhs.borrow_;
        isBorrowed_ = true;
      } else {
        own_ = rhs.own_;
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        borrow_ = rhs.borrow_;
      } else {
        new (&own_) T(rhs.own_);
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  MaybeOwned(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value)
  : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      borrow_ = rhs.borrow_;
    } else {
      new (&own_) T(std::move(rhs.own_));
    }
  }

  MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_assignable<T>::value) {
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
          own_.~T();
          borrow_ = rhs.borrow_;
          isBorrowed_ = true;
      } else {
        own_ = std::move(rhs.own_);
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        borrow_ = rhs.borrow_;
      } else {
        new (&own_) T(std::move(rhs.own_));
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  static MaybeOwned borrowed(const T& t) {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(in_place_t, Args&&... args) {
    return MaybeOwned(in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() {
    if (C10_UNLIKELY(!isBorrowed_)) {
      own_.~T();
    }
  }

  const T& operator*() const & {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(borrow_ != nullptr);
    }
    return C10_LIKELY(isBorrowed_) ? *borrow_ : own_;
  }

  const T* operator->() const {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(borrow_ != nullptr);
    }
    return C10_LIKELY(isBorrowed_) ? borrow_ : &own_;
  }

  // If borrowed, copy the underlying T. If owned, move from
  // it. borrowed/owned state remains the same, and either we
  // reference the same borrow as before or we are an owned moved-from
  // T.
  T operator*() && {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(borrow_ != nullptr);
      return *borrow_;
    } else {
      return std::move(own_);
    }
  }
};


} // namespace c10
