/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/unique_and_compact.h>

#include "./concurrent_id_hash_map.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& indices, const torch::Tensor unique_dst_ids) {
  torch::Tensor compacted_indices;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  torch::Tensor ids = torch::cat({unique_dst_ids, indices});
  AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "unique_and_compact", ([&] {
                               ConcurrentIdHashMap<scalar_t> id_map;
                               unique_ids = id_map.Init(ids, num_dst);
                               compacted_indices = id_map.MapIds(indices);
                             }));
  return std::tuple(unique_ids, compacted_indices);
}
}  // namespace sampling
}  // namespace graphbolt
