// Derived from PyTorch (c10/metal and aten/src/ATen/native/mps/kernels, BSD-3-Clause, see https://github.com/pytorch/pytorch):
// a standalone concatenation used only by the apps/attn_tune experiment.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_stdlib>

// Metal ports of the fp8 <-> fp32 conversions from
// torch/headeronly/util/Float8_*.h. Those headers cannot be included as-is
// in Metal shaders: they depend on the C++ standard library and use double.

namespace c10 {
namespace metal {
namespace detail {

inline float fp8e4m3fn_to_fp32(uchar input) {
  const uint w = uint(input) << 24;
  const uint sign = w & 0x80000000;
  const uint nonsign = w & 0x7fffffff;
  uint renorm_shift = nonsign == 0 ? 0 : ::metal::clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  const uint nan_mask = nonsign == 0x7f000000 ? 0x7f800000 : 0;
  const uint zero_mask = nonsign == 0 ? 0xffffffff : 0;
  const uint result = sign |
      ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
        nan_mask) &
       ~zero_mask);
  return as_type<float>(result);
}

inline uchar fp8e4m3fn_from_fp32(float value) {
  constexpr uint fp8_max = 1087u << 20;
  constexpr uint denorm_mask = 141u << 23;
  uint bits = as_type<uint>(value);
  const uint sign = bits & 0x80000000;
  bits ^= sign;
  uchar result;
  if (bits >= fp8_max) {
    result = bits > 0x7f800000 ? 0x7f : 0x7e;
  } else if (bits < (121u << 23)) {
    bits = as_type<uint>(as_type<float>(bits) + as_type<float>(denorm_mask));
    result = static_cast<uchar>(bits - denorm_mask);
  } else {
    const uchar mantissa_odd = (bits >> 20) & 1;
    bits += 0xc4000000 + 0x7ffff + mantissa_odd;
    result = static_cast<uchar>(bits >> 20);
    result = result == 0x7f ? 0x7e : result;
  }
  return result | static_cast<uchar>(sign >> 24);
}

} // namespace detail

struct alignas(1) float8_e4m3fn {
  uchar x;
  float8_e4m3fn() = default;
  template <typename T>
  float8_e4m3fn(T value) : x(detail::fp8e4m3fn_from_fp32(float(value))) {}
  operator float() const {
    return detail::fp8e4m3fn_to_fp32(x);
  }
};

template <typename T>
constexpr constant bool is_float8_v = ::metal::is_same_v<T, float8_e4m3fn>;

} // namespace metal
} // namespace c10
// Set of global constants that could be shareable between CPU and Metal code

#ifdef __METAL__
#include <metal_array>
#define C10_METAL_CONSTEXPR constant constexpr
#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define C10_METAL_HAS_MPP 1
#else
#define C10_METAL_HAS_MPP 0
#endif
#else
#include <array>
#define C10_METAL_CONSTEXPR constexpr
#endif

// `if IF_CONSTEXPR (...)` reduces to `if constexpr` on Metal 4 (which supports
// C++17) and to a plain runtime `if` on Metal 3 (where the compiler will
// usually still constant-fold the branch, but it's not guaranteed).
#if __METAL_VERSION__ >= 400
#define IF_CONSTEXPR constexpr
#else
#define IF_CONSTEXPR
#endif

#define C10_METAL_ALL_TYPES_FUNCTOR(_) \
  _(Byte, 0, uchar)                    \
  _(Char, 1, char)                     \
  _(Short, 2, short)                   \
  _(Int, 3, int)                       \
  _(Long, 4, long)                     \
  _(Half, 5, half)                     \
  _(Float, 6, float)                   \
  _(ComplexHalf, 8, half2)             \
  _(ComplexFloat, 9, float2)           \
  _(Bool, 11, bool)                    \
  _(BFloat16, 15, bfloat)              \
  _(Float8_e4m3fn, 24, float8_e4m3fn)  \
  _(UInt16, 27, uint16_t)              \
  _(UInt32, 28, uint32_t)              \
  _(UInt64, 29, uint64_t)

namespace c10 {
namespace metal {
C10_METAL_CONSTEXPR unsigned max_ndim = 16;
C10_METAL_CONSTEXPR unsigned simdgroup_size = 32;
// Number of elements each thread processes in dense elementwise kernels.
// Reading/writing a small array of values per thread improves memory-level
// parallelism vs. a one-element-per-thread kernel.
C10_METAL_CONSTEXPR unsigned ILP_PER_THREAD = 4;
// String form of ILP_PER_THREAD, used to embed the unroll width in kernel
// host names (e.g. `add_dense_ilp4_float_float`). Keep in sync with
// ILP_PER_THREAD; a static_assert in indexing.h enforces this.
#define C10_METAL_ILP_PER_THREAD_STR "4"

#ifdef __METAL__
template <typename T, unsigned N>
using array = ::metal::array<T, N>;
template <typename T>
using vec3 = ::metal::vec<T, 3>;
// Metal's builtin 2-component vectors (float2/half2) are the complex ABI.
template <typename T>
using complex = ::metal::vec<T, 2>;
#else
template <typename T, unsigned N>
using array = std::array<T, N>;
// Host mirror of Metal's 3-component vector ABI. Metal aligns int3/uint3 to
// 16 bytes, so alignas pads sizeof to 4 * sizeof(T) and mtl_setBytes uploads
// the full width the shader expects.
template <typename T>
struct alignas(4 * sizeof(T)) vec3 {
  T x, y, z;
};
// Host mirror of Metal's float2/half2 complex ABI. c10::complex<T> is
// alignas(2 * sizeof(T)), matching the layout of Metal's vec<T, 2>.
template <typename T>
using complex = ::c10::complex<T>;
#endif

// Integer ceiling division: ceil(a / b). Usable from both host code and
// Metal shaders (where the overload is selected by ADL via `using namespace
// c10::metal;` in shader sources).
template <typename T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// Round `a` up to the next multiple of `b`: ceil(a / b) * b.
template <typename T>
inline T round_up(T a, T b) {
  return ceil_div(a, b) * b;
}

enum class ScalarType {
#define _DEFINE_ENUM_VAL_(_v, _n, _t) _v = _n,
  C10_METAL_ALL_TYPES_FUNCTOR(_DEFINE_ENUM_VAL_)
#undef _DEFINE_ENUM_VAL_
};

} // namespace metal
} // namespace c10
// Metal helper functions
#include <metal_stdlib>

namespace c10 {
namespace metal {

namespace detail {
template <typename T>
struct vectypes {};

template <>
struct vectypes<float> {
  using type4 = float4;
  using type3 = float3;
  using type2 = float2;
};

template <>
struct vectypes<half> {
  using type4 = half4;
  using type3 = half3;
  using type2 = half2;
};

template <>
struct vectypes<bfloat> {
  using type4 = bfloat4;
  using type3 = bfloat3;
  using type2 = bfloat2;
};

template <>
struct vectypes<short> {
  using type4 = short4;
  using type3 = short3;
  using type2 = short2;
};

template <>
struct vectypes<int> {
  using type4 = int4;
  using type3 = int3;
  using type2 = int2;
};

template <>
struct vectypes<long> {
  using type4 = long4;
  using type3 = long3;
  using type2 = long2;
};

template <typename T>
struct OpMathType {
  using type = T;
};

template <>
struct OpMathType<half> {
  using type = float;
};

template <>
struct OpMathType<short> {
  using type = int;
};

template <>
struct OpMathType<char> {
  using type = int;
};

template <>
struct OpMathType<uchar> {
  using type = int;
};

template <>
struct OpMathType<bfloat> {
  using type = float;
};

template <>
struct OpMathType<half2> {
  using type = float2;
};

// Type promotion structure for higher precision accumulation
template <typename T>
struct AccumulationType {
  using type = T;
};

// Specialization for half - promote to float for accumulation
template <>
struct AccumulationType<half> {
  using type = float;
};

// Specialization for bfloat - promote to float for accumulation
template <>
struct AccumulationType<bfloat> {
  using type = float;
};

} // namespace detail

template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> max(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::max(a, b);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
max(T a, U b) {
  return ::metal::max(a, static_cast<T>(b));
}

template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> min(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::min(a, b);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
min(T a, U b) {
  return ::metal::min(a, static_cast<T>(b));
}

template <>
inline bfloat min(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::min(float(a), float(b)));
}

template <>
inline bfloat max(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::max(float(a), float(b)));
}

// Less-than that sorts NaNs last. Uses a != a instead of metal::isnan,
// which gave wrong results at large threadgroup sizes on M2.
template <typename T>
inline ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> less(
    T a,
    T b) {
  if (a != a)
    return false;
  if (b != b)
    return true;
  return a < b;
}

template <typename T>
inline ::metal::enable_if_t<!::metal::is_floating_point_v<T>, bool> less(
    T a,
    T b) {
  return a < b;
}

template <typename T>
using vec2type_t = typename detail::vectypes<T>::type2;

template <typename T>
using vec4type_t = typename detail::vectypes<T>::type4;

template <typename T>
using opmath_t = typename detail::OpMathType<T>::type;

template <typename T>
using accum_t = typename detail::AccumulationType<T>::type;

template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
  T v[N];
};

// TODO: Move it to type_traits header may be
template <typename F, typename... Args>
using result_of = decltype(::metal::declval<F>()(::metal::declval<Args>()...));

template <typename T>
constexpr constant bool is_complex_v =
    ::metal::is_same_v<T, float2> || ::metal::is_same_v<T, half2>;

template <typename T>
constexpr constant bool is_scalar_floating_point_v =
    ::metal::is_floating_point_v<T> && ::metal::is_scalar_v<T>;

template <typename T>
constexpr constant bool is_scalar_integral_v =
    ::metal::is_integral_v<T> && ::metal::is_scalar_v<T>;

template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));

// floor_divide
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  const auto quot = x / y;
  return (x < 0) == (y < 0) ? quot : (x % y != 0) ? quot - 1 : quot;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> && is_scalar_floating_point_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  return ::metal::floor(x / y);
}

// Python floor-division semantics (torch `rounding_mode="floor"`), a port of
// c10::div_floor_floating. Unlike floor_divide's naive floor(x/y), this handles
// non-finite and signed-zero inputs (e.g. inf // 1 -> nan, 1 // -inf -> -1).
template <
    typename T,
    ::metal::enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
inline T div_floor(const T a, const T b) {
  if (b == 0) {
    return a / b;
  }
  const auto mod = ::metal::fmod(a, b);
  auto div = (a - mod) / b;
  if (mod != 0 && (b < 0) != (mod < 0)) {
    div -= T(1);
  }
  if (div == 0) {
    return ::metal::copysign(T(0), a / b);
  }
  const auto floordiv = ::metal::floor(div);
  return div - floordiv > T(0.5) ? floordiv + T(1) : floordiv;
}

// Workaround for Metal compiler bug: the compiler produces wrong results
// when optimizing fused (x / A) % B expressions for integral types.
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> safe_mod(volatile T x, U y) {
  return x % y;
}

// fmod
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> fmod(T x, U y) {
  return x % y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> && is_scalar_floating_point_v<U>,
        bool> = true>
inline common_dtype<T, U> fmod(T x, U y) {
  return ::metal::fmod(x, y);
}

// cast_to primitives
//  - No-op if types as the same
template <
    typename T,
    typename U,
    ::metal::enable_if_t<::metal::is_same_v<U, T>, bool> = true>
inline T cast_to(const U from) {
  return from;
}
//  - Simple cast between scalar and complex dtypes
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        !::metal::is_same_v<U, T> && (is_complex_v<T> == is_complex_v<U>) &&
            !is_float8_v<T> && !is_float8_v<U>,
        bool> = true>
inline T cast_to(const U from) {
  return static_cast<T>(from);
}

// Metal rejects static_cast<bfloat>(fp8)
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        !::metal::is_same_v<U, T> && !is_complex_v<T> && !is_complex_v<U> &&
            (is_float8_v<T> || is_float8_v<U>),
        bool> = true>
inline T cast_to(const U from) {
  return T(float(from));
}

// - Scalar to complex
template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && !is_complex_v<U>, bool> = true>
inline T cast_to(const U from) {
  return T(float(from), 0.0);
}
// - Complex to bool: nonzero test over both components (matches bool(complex)).
template <
    typename T,
    typename U,
    ::metal::enable_if_t<::metal::is_same_v<T, bool> && is_complex_v<U>, bool> =
        true>
inline T cast_to(const U from) {
  return from.x != 0 || from.y != 0;
}
// - Complex to non-bool scalar (discards the imaginary part, matching CPU).
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        !is_complex_v<T> && !::metal::is_same_v<T, bool> && is_complex_v<U>,
        bool> = true>
inline T cast_to(const U from) {
  return static_cast<T>(from.x);
}

// Generalizable math operators (used for both scalar and complex)

template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return x * y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return T(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline common_dtype<T, U> div(const T x, const U y) {
  return x / y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline common_dtype<T, U> div(const T x, const U y) {
  // Purely real divisor: bypass the cross-term to avoid inf*0 / nan*0 = NaN
  // tainting the imag part. Mathematically equivalent for finite y.x.
  if (y.y == 0) {
    return T(x.x / y.x, x.y / y.x);
  }
  if (::metal::abs(y.x) >= ::metal::abs(y.y)) {
    const auto rat = y.y / y.x;
    return T(x.x + x.y * rat, x.y - x.x * rat) / (y.x + y.y * rat);
  }
  const auto rat = y.x / y.y;
  return T(x.x * rat + x.y, x.y * rat - x.x) / (y.y + y.x * rat);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline common_dtype<T, U> fma(
    const T x,
    const U y,
    const common_dtype<T, U> z) {
  return ::metal::fma(x, y, z);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline common_dtype<T, U> fma(
    const T x,
    const U y,
    const common_dtype<T, U> z) {
  return z + mul(x, y);
}

// Remainder operator
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> || is_scalar_floating_point_v<U>,
        bool> = true>
inline float remainder(const T x, const U y) {
  const auto x_f = static_cast<float>(x);
  const auto y_f = static_cast<float>(y);
  return x_f - y_f * floor_divide(x_f, y_f);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> remainder(const T x, const U y) {
  auto rc = x % y;
  return rc == 0 || (x ^ y) > 0 ? rc : rc + y;
}

// Based on aten/src/ATen/native/Pow.h
template <
    typename T,
    ::metal::enable_if_t<is_scalar_integral_v<T>, bool> = true>
inline T powi_impl(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
      result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> || is_scalar_floating_point_v<U>,
        bool> = true>
inline float pow(T a, U b) {
  return ::metal::precise::pow(static_cast<float>(a), static_cast<float>(b));
}

// Complex pow - use polar form: a = r*e^(i*theta)
// a^b = exp(b * log(a)) = exp(b * (log(r) + i*theta))
template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline float2 pow(T a, U b) {
  // Convert a to polar form
  // Use explicit computation instead of length() due to numerical issues
  const auto r = ::metal::precise::sqrt(a.x * a.x + a.y * a.y);

  // Special case: 0^b is 0 if Re(b) > 0, undefined otherwise
  if (r == 0.0) {
    return b.x > 0 ? float2(0.0, 0.0) : float2(NAN, NAN);
  }

  const auto theta = ::metal::precise::atan2(a.y, a.x);
  const auto log_r = ::metal::precise::log(r);

  // Calculate a^b = r^b * e^(i*theta*b)
  // new_r = exp(b.x * log(r) - b.y * theta)
  // new_theta = b.x * theta + b.y * log(r)
  const auto new_r = ::metal::precise::exp(b.x * log_r - b.y * theta);
  const auto new_theta = b.x * theta + b.y * log_r;

  return float2(
      new_r * ::metal::precise::cos(new_theta),
      new_r * ::metal::precise::sin(new_theta));
}

// Integral pow - unsigned types
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && !::metal::is_signed_v<T>,
        bool> = true>
inline T pow(T a, U b) {
  return powi_impl(a, T(b));
}

// Integral pow - signed types
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T>&& ::metal::is_signed_v<T>,
        bool> = true>
inline T pow(T a, U b) {
  if (b < 0) {
    if (a == 1) {
      return 1;
    } else if (a == -1) {
      auto negative = (-b) % static_cast<T>(2);
      return negative ? -1 : 1;
    } else {
      return 0;
    }
  }
  return powi_impl(a, T(b));
}

// Based on algorithm described in
// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  const auto xp1 = 1.0f + x;
  // First two elements of Taylor series for log(1+x) in Horner's form are:
  // log(1+x) = x * (1 - x * (.5 ...)), but if 1 + x == x, then it's just x
  if (xp1 == 1.0f) {
    return x;
  }
  auto rc = ::metal::precise::log(xp1);
  if (x > -.5 && x < .5) {
    // Order of operations is important here for higher precision
    rc *= x / (xp1 - 1.0f);
  }
  return rc;
}

// The function is ported from mlx
inline float2 log1p(float2 in) {
  float x = in.x;
  float y = in.y;
  float zabs = ::metal::precise::sqrt(x * x + y * y);
  float theta = ::metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = ::metal::sqrt((x + 1) * (x + 1) + y * y);
    return {::metal::log(z0), theta};
  }
}

template <typename T1, typename T2 = T1>
struct pair {
  T1 first;
  T2 second;
};

template <typename T>
inline T conj(T a) {
  return a;
}

template <>
inline half2 conj(half2 a) {
  return half2(a.x, -a.y);
}

template <>
inline float2 conj(float2 a) {
  return float2(a.x, -a.y);
}

// The following implementation of hypot provides better numerical stability
// than the naive implementation. It is based on:
// https://github.com/pearu/functional_algorithms/blob/7dbbfd7db225b1c202e0e364fc435423ccf52dbe/functional_algorithms/algorithms.py#L168
//
// This implementation changes the naive formula for the hypotenuse of a right
// triangle, `h = sqrt(a^2 + b^2)`, into three alternate forms to be used in
// different cases. The reason why the naive formula is unstable is because of
// the square terms. If `a` or `b` are very large or very small floating point
// numbers, then their squares will resolve to inf or 0.
//
// Assume `a >= b >= 0`. We can first change the formula to:
// `h = a sqrt(1 + (b / a)^2)`
// `h = a sqrt(1 + r)`
// where `r = (b / a)^2`. Since `a >= b >= 0`, then `1 >= r >= 0`.
//
// Case 0: Either input is inf
//  Return inf
//
// Case 1: `a == b`
//   The formula simplifies to `h = a sqrt(2)`.
//
// Case 2: `1 >> r > 0`
//   Due to floating point error, `sqrt(1 + r)` resolves to 1. So we use the
//   binomial approximation `sqrt(1 + r) ≈ 1 + r / 2`, and the formula becomes
//   `h ≈ a + a r / 2`.
//
// Case 3: All other cases.
//   Use `h = a sqrt(1 + r)`.
inline float hypot(float a_, float b_) {
  if (::metal::isinf(a_) || ::metal::isinf(b_)) {
    return INFINITY;
  }
  auto a = max(a_, b_);
  auto b = min(a_, b_);

  auto b_over_a = c10::metal::div(b, a);
  auto r = c10::metal::mul(b_over_a, b_over_a);
  auto sqrt_1_plus_r = ::metal::precise::sqrt(1 + r);

  auto h1 = M_SQRT2_F * a;
  auto h2 = a + a * r / 2;
  auto h3 = a * sqrt_1_plus_r;
  bool is_h1 = (a == b);
  bool is_h2 = ((sqrt_1_plus_r == 1) && (r > 0));

  return ::metal::select(::metal::select(h3, h2, is_h2), h1, is_h1);
}

#define INSTANTIATE_FOR_ALL_TYPES(MACRO) \
  MACRO(float);                          \
  MACRO(half);                           \
  MACRO(bfloat);                         \
  MACRO(float2);                         \
  MACRO(long);                           \
  MACRO(char);                           \
  MACRO(uchar);                          \
  MACRO(short);                          \
  MACRO(int);

#define INSTANTIATE_FOR_FLOAT_TYPES(MACRO) \
  MACRO(float);                            \
  MACRO(half);                             \
  MACRO(bfloat);

// Copy n bytes from src to dst using the widest vector their combined alignment
// allows (uint4 -> uint2 -> uint -> ushort -> byte), with a scalar tail.
inline void copy_bytes_aligned(
    device uchar* dst,
    device const uchar* src,
    uint n) {
  const uint align =
      (reinterpret_cast<ulong>(dst) | reinterpret_cast<ulong>(src)) & 15;
  uint i = 0;
  if (align == 0) {
    for (; i + 16 <= n; i += 16) {
      *reinterpret_cast<device uint4*>(dst + i) =
          *reinterpret_cast<device const uint4*>(src + i);
    }
  } else if ((align & 7) == 0) {
    for (; i + 8 <= n; i += 8) {
      *reinterpret_cast<device uint2*>(dst + i) =
          *reinterpret_cast<device const uint2*>(src + i);
    }
  } else if ((align & 3) == 0) {
    for (; i + 4 <= n; i += 4) {
      *reinterpret_cast<device uint*>(dst + i) =
          *reinterpret_cast<device const uint*>(src + i);
    }
  } else if ((align & 1) == 0) {
    for (; i + 2 <= n; i += 2) {
      *reinterpret_cast<device ushort*>(dst + i) =
          *reinterpret_cast<device const ushort*>(src + i);
    }
  }
  for (; i < n; ++i) {
    dst[i] = src[i];
  }
}

} // namespace metal
} // namespace c10

using namespace metal;
// Flash-attention prefill kernel for MPS.
// Adapted from kernels-community/metal-flash-sdpa, which is itself
// adapted from MLX.
// Uses simdgroup-matrix MMA for high throughput; supports causal mask,
// optional additive/bool mask, GQA/MQA and softcapping.
// Operates on PyTorch's dense [B, H, L, D] layout via per-tensor strides.
//
// Included from Attention.metal — no top-level includes / `using` here so the
// includer controls them. Builds as part of Attention.metallib (one file →
// one bundled section), so Attention.mm only needs a single library handle.


#define PREFILL_CONST static constant constexpr const
#define PREFILL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
struct pointer_element {};
template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

template <int val>
using Int = integral_constant<int, val>;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  PREFILL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  PREFILL_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device T* src;

  METAL_FUNC BlockLoaderT(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) {
    // c10::metal::max propagates NaN (plain max drops it -> row max -inf -> 0).
    return c10::metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  // If y (the row max) is -inf, every score in this row was masked out
  // (e.g. an explicit additive mask of all -inf, or a fully causally-masked
  // row). exp(-inf - -inf) = exp(NaN) = NaN, which then poisons the running
  // sum and output. Returning 0 is the mathematically correct limit
  // (exp(-inf) = 0) and is what flash-attention implementations do.
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -INFINITY) ? T(0) : fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(kFragRows_ == 8, "Only 8x8 fragments are supported");
  static_assert(kFragCols_ == 8, "Only 8x8 fragments are supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  PREFILL_CONST int kFragRows = 8;
  PREFILL_CONST int kFragCols = 8;

  PREFILL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  PREFILL_CONST int kElemRows = 1;
  PREFILL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void load(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] = static_cast<T>(
              src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void store(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] =
            static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  PREFILL_CONST int kFragRows = MMAFrag_t::kFragRows;
  PREFILL_CONST int kFragCols = MMAFrag_t::kFragCols;
  PREFILL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  PREFILL_CONST int kTileRows = kTileRows_;
  PREFILL_CONST int kTileCols = kTileCols_;

  PREFILL_CONST int kRows = kTileRows * kFragRows;
  PREFILL_CONST int kCols = kTileCols * kFragCols;

  PREFILL_CONST int kNumFrags = kTileRows * kTileCols;
  PREFILL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  PREFILL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  PREFILL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const {
    PREFILL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  PREFILL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    PREFILL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      PREFILL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}

struct PrefillAttnParams {
  int B; // batch size
  int H; // number of query heads
  int D; // head dim
  int qL; // query sequence length
  int kL; // key sequence length
  int gqa_factor; // num_q_heads / num_kv_heads
  float scale;
  float softcapping;
  // Strides (B, H, L) - element stride in last dim is assumed to be 1.
  // 64-bit: a batch/head stride is a product of the other dimensions and
  // overflows 32 bits on large inputs, and the mask qL stride is multiplied
  // by an absolute query position, reaching qL * kL.
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

struct PrefillAttnMaskParams {
  int64_t M_strides[4]; // (B, H, qL, kL)
};

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    bool HAS_MASK,
    bool DO_CAUSAL,
    typename MaskType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
prefill_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant PrefillAttnParams* params [[buffer(4)]],
    const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  using AccumType = c10::metal::accum_t<T>;
  constexpr bool has_mask = HAS_MASK;
  constexpr bool do_causal = DO_CAUSAL;

  const int batch_idx = tid.z;
  const int head_idx = tid.y;
  const int block_idx = tid.x;

  const int q_seq_len = params->qL;
  const int k_seq_len = params->kL;

  if (block_idx * BQ >= q_seq_len) {
    return;
  }

  const ulong kv_head_idx = head_idx / params->gqa_factor;

  // Stride-based pointer offsets for [B, H, L, D] layout.
  Q += batch_idx * params->Q_strides[0] + head_idx * params->Q_strides[1] +
      (block_idx * BQ) * params->Q_strides[2];
  K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += batch_idx * params->O_strides[0] + head_idx * params->O_strides[1] +
      (block_idx * BQ) * params->O_strides[2];

  if IF_CONSTEXPR (has_mask) {
    mask += batch_idx * mask_params->M_strides[0] +
        head_idx * mask_params->M_strides[1];
  }

  // Threadgroup memory (with padding to avoid bank conflicts).
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Q / K / V loaders. K is loaded transposed.
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

  // Stride between consecutive sequence rows for Q/K/V/O.
  // Narrowed back to 32-bit: these only ever multiply in-tile offsets, the
  // base pointers having already been advanced.
  const int q_seq_stride = int(params->Q_strides[2]);
  const int k_seq_stride = int(params->K_strides[2]);
  const int v_seq_stride = int(params->V_strides[2]);
  const int o_seq_stride = int(params->O_strides[2]);

  QBlockLoader loader_q(Q, q_seq_stride, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, k_seq_stride, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, v_seq_stride, Vs, simd_group_id, simd_lane_id);

  // Apply softcapping by scaling the inputs before tanh.
  float adjusted_scale = params->scale;
  if (params->softcapping != 1.0f) {
    adjusted_scale = params->scale / params->softcapping;
  }
  // 1.44269504089 = 1 / ln(2) so we can use exp2 instead of exp. Scale the
  // scores in AccumType, not Q in T: rounding the scale and the scaled Q to
  // the input dtype compounds into a visible error at large scales.
  const AccumType scale2 = adjusted_scale * 1.44269504089f;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q block (with safe load if last block is partial).
  const int q_block_end = min(block_idx * BQ + BQ, q_seq_len);
  const int q_block_size = q_block_end - block_idx * BQ;

  if (q_block_size < BQ) {
    loader_q.load_safe(short2(BD, q_block_size));
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  PREFILL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -INFINITY;
  }

  // Causal masking can shorten the K range. PyTorch convention is upper-left
  // alignment: query at row r attends to keys at cols 0..r, regardless of qL
  // vs kL. So the last query in this block (row block_idx*BQ + q_block_size-1)
  // attends to cols 0..(block_idx*BQ + q_block_size - 1).
  int kb_lim = c10::metal::ceil_div(k_seq_len, BK);
  if IF_CONSTEXPR (do_causal) {
    int max_col = block_idx * BQ + q_block_size; // exclusive upper bound
    kb_lim = min(kb_lim, c10::metal::ceil_div(max_col, BK));
  }

  for (int kb = 0; kb < kb_lim; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int k_block_end = min(kb * BK + BK, k_seq_len);
    const int k_block_size = k_block_end - kb * BK;

    if (k_block_size < BK) {
      loader_k.load_safe(short2(BD, k_block_size));
    } else {
      loader_k.load_unsafe();
    }

    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    PREFILL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    {
      using stile_t = decltype(Stile);
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            Stile.frag_at(i, j)[jj] *= scale2;
          }
        }
      }
    }

    // Mask out partial K block.
    if (k_block_size < BK) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = -INFINITY;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= k_block_size) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Causal mask (PyTorch upper-left convention: row r sees cols 0..r).
    if IF_CONSTEXPR (do_causal) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = -INFINITY;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        int row_pos = block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Optional additive / boolean mask.
    if IF_CONSTEXPR (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -INFINITY;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos_in_seq =
            block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos_in_seq = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              mask_params->M_strides[2],
              Int<1>{},
              q_seq_len,
              k_seq_len,
              row_pos_in_seq,
              col_pos_in_seq);

          constexpr selem_t kLog2E = selem_t(1.44269504089f);
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if IF_CONSTEXPR (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += selem_t(mfrag[jj]) * kLog2E;
            }
          }
        }
      }
    }

    // Softcapping (tanh-based).
    if (params->softcapping != 1.0f) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      const selem_t softcapping_val = static_cast<selem_t>(params->softcapping);

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            Stile.frag_at(i, j)[jj] =
                metal::tanh(Stile.frag_at(i, j)[jj]) * softcapping_val;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k_block_size < BK) {
      loader_v.load_safe(short2(BD, k_block_size));
    } else {
      loader_v.load_unsafe();
    }

    // Online softmax update.
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // If new_max is -inf, both old/new max are -inf (nothing valid yet).
    // Use factor=1 so we keep the running state unchanged for that row.
    // exp(-inf - -inf) would otherwise produce NaN.
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (new_max[i] == -INFINITY)
          ? AccumType(1)
          : fast::exp2(max_score[i] - new_max[i]);
    }

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    PREFILL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      PREFILL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        PREFILL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if IF_CONSTEXPR (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if IF_CONSTEXPR (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    loader_k.next();
    loader_v.next();
  }
  PREFILL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    if (max_score[i] == -INFINITY) {
      sum_score[i] = AccumType(1);
    }
  }

  // Normalize and store output.
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  device T* O_tile = O + (tm + sm) * o_seq_stride + sn;

  if (q_block_size < BQ) {
    if ((tm + sm) < q_block_size && sn < BD) {
      auto dst_tile_dims = short2(BD - sn, q_block_size - (tm + sm));
      Otile.template store_safe<T, 1, 1>(O_tile, o_seq_stride, dst_tile_dims);
    }
  } else {
    Otile.template store<T, 1, 1>(O_tile, o_seq_stride);
  }
}

#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
// MPP cooperative-tensor flash-attention prefill for MPS, on
// mpp::tensor_ops::matmul2d. Adapted from MLX. Emitted only with the
// Metal 4 SDK; runtime dispatch gates on macOS 26.2+.


#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define MPP_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define MPP_CONST static constant constexpr const

// MSL's metal::integral_constant lacks arithmetic overloads, so add the
// minimum needed to multiply runtime indices by Int<1>{}
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(U lhs, metal::integral_constant<T, Val>) {
  return lhs * Val;
}
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(metal::integral_constant<T, Val>, U rhs) {
  return Val * rhs;
}

// 16x16 simdgroup tile, 8 cells/lane in 2 rows x 4 cols (rows fm, fm+8).
// Within-row col pattern is contig-quad {0,1,2,3} on Apple10 and split-pair
// {0,1,8,9} on Apple8, this matches what matmul2d::run() expects in each
// arch's ct_a/ct_b/ct_c slots so the mma below can register-copy directly.
template <bool IS_APPLE10>
struct BaseMPPFrag {
  MPP_CONST short kFragRows = 16;
  MPP_CONST short kFragCols = 16;
  MPP_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32; // 8
  MPP_CONST short kElemRows = 2;
  MPP_CONST short kElemCols = 4;
  MPP_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MPPFrag shape inconsistent with size");

  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    if (IS_APPLE10) {
      const short qid = lane >> 2;
      const short fm = (qid & 4) | ((lane >> 1) & 3);
      const short fn = ((qid & 2) | (lane & 1)) * 4;
      return short2{fn, fm};
    } else {
      const short fm = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3);
      const short fn = ((lane >> 3) & 1) * 4 + (lane & 1) * 2;
      return short2{fn, fm};
    }
  }

  // Slot j -> col offset: contig {0,1,2,3} on A10, split-pair {0,1,8,9} on A8.
  METAL_FUNC static constexpr short col_offset(short j) {
    if (IS_APPLE10) {
      return j;
    } else {
      return (j & 1) | ((j >> 1) << 3);
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + c + col_offset(j)]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + col_offset(j)) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + c + col_offset(j)]);
          }
        } else {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + col_offset(j)) * str_y]);
          }
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        const auto coff = col_offset(j);
        if (r < lx && (c + coff) < ly) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + coff) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + col_offset(j)] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + col_offset(j)) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + col_offset(j)] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + col_offset(j)) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  // C[16,32] += A[16,16] * [Bn0|Bn1] (Bn0/Bn1 packed along N).
  // ct_b/ct_c slot order: A10 = (Bn0_0..7 | Bn1_0..7);
  // A8 = (Bn0[0..3] | Bn1[0..3] | Bn0[4..7] | Bn1[4..7]).
  // For Q@K^T (transpose_b=true), Bn0/Bn1 come from BaseMPPFragK so the
  // per-lane geometry already matches; slot copies are identical for both
  // transpose modes.
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    if (IS_APPLE10) {
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = Bn0[i];
        ct_b[kElemsPerFrag + i] = Bn1[i];
      }
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_c[i] = Cn0[i];
        ct_c[kElemsPerFrag + i] = Cn1[i];
      }

      gemm_op.run(ct_a, ct_b, ct_c);

      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        Cn0[i] = ct_c[i];
        Cn1[i] = ct_c[kElemsPerFrag + i];
      }
    } else {
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        ct_b[i] = Bn0[i];
        ct_b[4 + i] = Bn1[i];
        ct_b[8 + i] = Bn0[4 + i];
        ct_b[12 + i] = Bn1[4 + i];
      }
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        ct_c[i] = Cn0[i];
        ct_c[4 + i] = Cn1[i];
        ct_c[8 + i] = Cn0[4 + i];
        ct_c[12 + i] = Cn1[4 + i];
      }

      gemm_op.run(ct_a, ct_b, ct_c);

      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        Cn0[i] = ct_c[i];
        Cn1[i] = ct_c[4 + i];
        Cn0[4 + i] = ct_c[8 + i];
        Cn1[4 + i] = ct_c[12 + i];
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

// Second operand of Q@K^T (transpose_b=true) on Apple8: 4x2 split-pair per
// lane, rows {fm0, fm0+1, fm0+8, fm0+9}, cols {fn0, fn0+8}. K is read-only
// here, so only load / load_rows are implemented. On Apple10 the regular
// BaseMPPFrag layout already matchess
struct BaseMPPFragK_Apple8 {
  MPP_CONST short kFragRows = 16;
  MPP_CONST short kFragCols = 16;
  MPP_CONST short kElemsPerFrag = 8;
  // Rows aren't equally spaced ({fm0, fm0+1, fm0+8, fm0+9}); kElemRowsJump
  // is unused. kElemRows/kElemCols exist because MPPTile reads them.
  MPP_CONST short kElemRows = 4;
  MPP_CONST short kElemCols = 2;
  MPP_CONST short kElemRowsJump = 1;

  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    const short fm0 = ((lane >> 3) & 1) * 4 + (lane & 1) * 2;
    const short fn0 = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3);
    return short2{fn0, fm0};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short s = 0; s < kElemsPerFrag; s++) {
      const short s_local = s & 3;
      const short r = off_x + (s_local & 1) + 8 * ((s_local >> 1) & 1);
      const short c = off_y + 8 * (s >> 2);
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        dst[s] = static_cast<T>(src[r * str_x + c]);
      } else {
        dst[s] = static_cast<T>(src[r * str_x + c * str_y]);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    MPP_PRAGMA_UNROLL
    for (short s = 0; s < kElemsPerFrag; s++) {
      const short s_local = s & 3;
      const short r = off_x + (s_local & 1) + 8 * ((s_local >> 1) & 1);
      const short c = off_y + 8 * (s >> 2);
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          dst[s] = static_cast<T>(src[r * str_x + c]);
        } else {
          dst[s] = static_cast<T>(src[r * str_x + c * str_y]);
        }
      } else {
        dst[s] = T(0);
      }
    }
  }
};

template <bool IS_APPLE10>
using BaseMPPFragK =
    metal::conditional_t<IS_APPLE10, BaseMPPFrag<true>, BaseMPPFragK_Apple8>;

template <typename T, short kTileRows_, short kTileCols_, class MPPFrag_>
struct MPPTile {
  using MPPFrag_t = MPPFrag_;
  using elem_type = T;

  MPP_CONST short kFragRows = MPPFrag_t::kFragRows;
  MPP_CONST short kFragCols = MPPFrag_t::kFragCols;
  MPP_CONST short kElemsPerFrag = MPPFrag_t::kElemsPerFrag;

  MPP_CONST short kTileRows = kTileRows_;
  MPP_CONST short kTileCols = kTileCols_;

  MPP_CONST short kRows = kTileRows * kFragRows;
  MPP_CONST short kCols = kTileCols * kFragCols;

  MPP_CONST short kNumFrags = kTileRows * kTileCols;
  MPP_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  MPP_CONST short kFragThrRows = MPPFrag_t::kElemRows;
  MPP_CONST short kFragThrCols = MPPFrag_t::kElemCols;
  MPP_CONST short kFragRowsJump = MPPFrag_t::kElemRowsJump;

  MPP_CONST short kRowsPerThread = kTileRows * MPPFrag_t::kElemRows;
  MPP_CONST short kColsPerThread = kTileCols * MPPFrag_t::kElemCols;

  using frag_type = typename MPPFrag_t::template dtype_frag_t<T>;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MPPTile() thread {}

  METAL_FUNC constexpr void clear() {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }
  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::load(
            frag_at(i, j), src, ld, Int<1>{}, i * kFragRows, j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_rows(
      const device U* src,
      const int ld,
      const short n_rows) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::load_rows(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            n_rows,
            i * kFragRows,
            j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::store(
            frag_at(i, j), dst, ld, Int<1>{}, i * kFragRows, j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::store_rows(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            n_rows,
            i * kFragRows,
            j * kFragCols);
      }
    }
  }
};

// Flash-attention prefill via matmul2d. Mirrors prefill_attention's contract
// (PrefillAttnParams + 4D mask, last-dim contiguous, [B, H, L, D] layout,
// upper-left causal). One threadgroup per (Q-block, head, batch) tile.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    bool HAS_MASK,
    bool DO_CAUSAL,
    bool IS_APPLE10,
    typename MaskType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
prefill_attention_mpp(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant PrefillAttnParams* params [[buffer(4)]],
    const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using AccumType = c10::metal::accum_t<T>;

  const int batch_idx = tid.z;
  const int head_idx = tid.y;
  const int block_idx = tid.x;

  const int qL = params->qL;
  const int kL = params->kL;

  if (block_idx * BQ >= qL) {
    return;
  }

  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const bool align_K = (kL_rem == 0);
  const int NK = align_K ? NK_aligned : (NK_aligned + 1);

  const int NQ_aligned = qL / BQ;
  const int qL_rem = qL - NQ_aligned * BQ;
  const bool align_Q = (qL_rem == 0);

  const int kv_head_idx = head_idx / params->gqa_factor;

  Q += batch_idx * params->Q_strides[0] + head_idx * params->Q_strides[1] +
      (block_idx * BQ) * params->Q_strides[2];
  K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += batch_idx * params->O_strides[0] + head_idx * params->O_strides[1] +
      (block_idx * BQ) * params->O_strides[2];

  if constexpr (HAS_MASK) {
    mask += batch_idx * mask_params->M_strides[0] +
        head_idx * mask_params->M_strides[1];
  }

  // Narrowed back to 32-bit: these only ever multiply in-tile offsets, the
  // base pointers having already been advanced.
  const int q_seq_stride = int(params->Q_strides[2]);
  const int k_seq_stride = int(params->K_strides[2]);
  const int v_seq_stride = int(params->V_strides[2]);
  const int o_seq_stride = int(params->O_strides[2]);

  // Pre-multiply by 1/ln(2) so we can use exp2 in the inner softmax.
  const AccumType scale2 =
      static_cast<AccumType>(params->scale) * 1.44269504089f;

  constexpr short kU = 16;
  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host at least 1 MPP fragment along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kU);
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  static_assert(TQ == 1, "Check TQ");
  static_assert(
      (TK & 1) == 0,
      "TK must be even: each Q@K mma writes 2 N-fragments at a time.");
  static_assert(
      (TD & 1) == 0,
      "TD must be even: each P@V mma writes 2 N-fragments at a time.");

  using QFrag_t = BaseMPPFrag<IS_APPLE10>;
  using KFrag_t = BaseMPPFragK<IS_APPLE10>;
  using otile_t = MPPTile<AccumType, TQ, TD, QFrag_t>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * simd_group_id;
  Q += tm * q_seq_stride;

  const short2 simd_coord = otile_t::MPPFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -INFINITY;
  }

  // PyTorch upper-left causal: query row r attends to key cols 0..r,
  // independent of qL vs kL.
  int kb_lim = NK;
  int kb_min_causal = NK;
  if constexpr (DO_CAUSAL) {
    int q_max = (block_idx + 1) * BQ;
    kb_lim = min(NK, (q_max + BK - 1) / BK);
    int q_min = block_idx * BQ;
    kb_min_causal = q_min / BK;
  }

  const bool is_last_q = (block_idx == NQ_aligned);
  const short lim_rows_q = qL_rem - tm;
  const short lim_rows_k = kL_rem;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = (kb == NK_aligned);

    using stile_t = MPPTile<AccumType, TQ, TK, QFrag_t>;
    stile_t Stile;
    Stile.clear();

    // S = Q @ K^T. K is loaded row-major and consumed transposed by mma.
    MPP_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      MPP_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        MPP_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          MPPTile<T, 1, 1, QFrag_t> Qtile;
          MPPTile<T, 2, 1, KFrag_t> Ktile;

          const int Q_load_off = iq * kU * q_seq_stride + id * kU;
          const int K_load_off = ik * kU * k_seq_stride + id * kU;

          if (!align_Q && is_last_q) {
            Qtile.load_rows(Q + Q_load_off, q_seq_stride, lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, q_seq_stride);
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(K + K_load_off, k_seq_stride, lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, k_seq_stride);
          }

          QFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale (in log2 space).
    MPP_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    // Mask out the partial K column when kL is not BK-aligned.
    if (!align_K && is_last_k) {
      MPP_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          MPP_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            MPP_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              const short c_off = QFrag_t::col_offset(jj);
              if ((col_pos + c_off) >= kL_rem) {
                fg[loc] = -INFINITY;
              }
            }
          }
        }
      }
    }

    // Causal mask (upper-left).
    if constexpr (DO_CAUSAL) {
      if (kb >= kb_min_causal) {
        const int base_row = block_idx * BQ + tm;
        const int base_col = kb * BK;
        MPP_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          MPP_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            thread auto& fg = Stile.frag_at(iq, ik);
            MPP_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
              MPP_PRAGMA_UNROLL
              for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                const int r =
                    base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
                const int c = base_col + ik * kU + QFrag_t::col_offset(jj) + sn;
                const auto loc = ii * stile_t::kFragThrCols + jj;
                if (r < c) {
                  fg[loc] = -INFINITY;
                }
              }
            }
          }
        }
      }
    }

    // Optional additive / boolean attention mask.
    if constexpr (HAS_MASK) {
      constexpr bool is_bool = metal::is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = MPPTile<melem_t, TQ, TK, QFrag_t>;
      using mfrag_t = typename mtile_t::frag_type;

      const int base_row = block_idx * BQ + tm;
      const int base_col = kb * BK;
      const bool inside = (base_row + BQ <= qL) && (base_col + BK <= kL);

      MPP_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row + iq * kU;
          const int col_pos = base_col + ik * kU;
          mfrag_t mfrag;
          if (inside) {
            mtile_t::MPPFrag_t::load(
                mfrag,
                mask,
                mask_params->M_strides[2],
                Int<1>{},
                row_pos,
                col_pos);
          } else {
            mtile_t::MPPFrag_t::load_safe(
                mfrag,
                mask,
                mask_params->M_strides[2],
                Int<1>{},
                qL,
                kL,
                row_pos,
                col_pos);
          }
          thread auto& fg = Stile.frag_at(iq, ik);
          MPP_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : (AccumType)-INFINITY;
            } else {
              fg[jj] += 1.44269504089f * static_cast<AccumType>(mfrag[jj]);
            }
          }
        }
      }
    }

    // Online softmax update.
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // If new_max is -inf, the row is fully masked so far; keep state.
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (new_max[i] == -INFINITY)
          ? (AccumType)1
          : fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O += P @ V.
    MPP_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      MPP_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          MPPTile<T, 1, 2, QFrag_t> Vtile;
          const int V_load_off = ik * kU * v_seq_stride + id * kU;
          if (!align_K && is_last_k) {
            Vtile.load_rows(V + V_load_off, v_seq_stride, lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, v_seq_stride);
          }
          QFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += BK * k_seq_stride;
    V += BK * v_seq_stride;
  }

  // Drains the cooperative-tensor pipeline before the O registers are read
  // back below.
  threadgroup_barrier(mem_flags::mem_none);

  // Treat fully-masked rows as zero output.
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    if (max_score[i] == -INFINITY) {
      sum_score[i] = (AccumType)1;
    }
  }
  metal::vec<AccumType, kRowsPT> rcp;
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.0f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  O += tm * o_seq_stride;
  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0) {
      return;
    }
    Otile.store_rows(O, o_seq_stride, lim_rows_q);
  } else {
    Otile.store(O, o_seq_stride);
  }
}

#define instantiate_mpp_attn_arch(                                        \
    arch_suffix,                                                          \
    is_a10,                                                               \
    tname,                                                                \
    dtype,                                                                \
    bq,                                                                   \
    bk,                                                                   \
    bd,                                                                   \
    wm,                                                                   \
    wn,                                                                   \
    hm,                                                                   \
    dc,                                                                   \
    mname,                                                                \
    mtype)                                                                \
  instantiate_kernel(                                                     \
      "prefill_attention_mpp_" arch_suffix "_" #tname "_bq" #bq "_bk" #bk \
      "_bd" #bd "_wm" #wm "_wn" #wn "_hm" #hm "_dc" #dc "_mask" #mname,   \
      prefill_attention_mpp,                                              \
      dtype,                                                              \
      bq,                                                                 \
      bk,                                                                 \
      bd,                                                                 \
      wm,                                                                 \
      wn,                                                                 \
      hm,                                                                 \
      dc,                                                                 \
      is_a10,                                                             \
      mtype)

#define instantiate_mpp_attn(                                               \
    tname, dtype, bq, bk, bd, wm, wn, hm, dc, mname, mtype)                 \
  instantiate_mpp_attn_arch(                                                \
      "a10", true, tname, dtype, bq, bk, bd, wm, wn, hm, dc, mname, mtype); \
  instantiate_mpp_attn_arch(                                                \
      "a9", false, tname, dtype, bq, bk, bd, wm, wn, hm, dc, mname, mtype);

// BK is always 32: mpp_prefill_block_shape_for_head_dim returns it for every
// supported head dim.
#define instantiate_mpp_attn_shapes(iname, itype, hm, dc, mname, mtype)        \
  instantiate_mpp_attn(iname, itype, 64, 32, 256, 4, 1, hm, dc, mname, mtype); \
  instantiate_mpp_attn(iname, itype, 64, 32, 128, 4, 1, hm, dc, mname, mtype); \
  instantiate_mpp_attn(iname, itype, 64, 32, 96, 4, 1, hm, dc, mname, mtype);  \
  instantiate_mpp_attn(iname, itype, 64, 32, 64, 4, 1, hm, dc, mname, mtype);

#define instantiate_mpp_attn_causal(iname, itype, hm, mname, mtype) \
  instantiate_mpp_attn_shapes(iname, itype, hm, 0, mname, mtype);   \
  instantiate_mpp_attn_shapes(iname, itype, hm, 1, mname, mtype);

// A bool mask name is only reachable with hm=1: the host leaves the mask dtype
// at the query dtype when there is no mask.
#define instantiate_mpp_attn_mask(iname, itype)               \
  instantiate_mpp_attn_causal(iname, itype, 0, iname, itype); \
  instantiate_mpp_attn_causal(iname, itype, 1, iname, itype); \
  instantiate_mpp_attn_causal(iname, itype, 1, bool_, bool);

#else
#error no MPP
#endif
