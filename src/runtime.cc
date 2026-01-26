#include "helios/runtime.h"

namespace helios {

RunResult Runtime::run(const Operator&, Scheduler&, real_t*, const RuntimeConfig&) {
  return {};
}

real_t Runtime::residual_inf(const Operator&, const real_t*, int) {
  return 0.0;
}

RunResult Runtime::run_jacobi_(const Operator&, real_t*, const RuntimeConfig&) {
  return {};
}

RunResult Runtime::run_gauss_seidel_(const Operator&, real_t*, const RuntimeConfig&) {
  return {};
}

RunResult Runtime::run_async_(const Operator&, Scheduler&, real_t*, const RuntimeConfig&) {
  return {};
}

} // namespace helios