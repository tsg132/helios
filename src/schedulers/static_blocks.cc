#include "helios/schedulers/static_blocks.h"

namespace helios {

void StaticBlocksScheduler::init(index_t n, size_t num_threads) {
  n_ = n;
  num_threads_ = (num_threads > 0) ? num_threads : 1;
}

index_t StaticBlocksScheduler::next(size_t /*tid*/) {
  // Stub: always return 0 so the project links.
  // We'll implement real static block stepping next.
  return 0;
}

} // namespace helios