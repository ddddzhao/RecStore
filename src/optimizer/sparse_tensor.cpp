#include "sparse_tensor.h"

uint64_t SparseTensor::concatKeyAndDim(uint64_t key, DIM_TYPE dim) {
  key &= 0x00FFFFFFFFFFFFFF;
  return (static_cast<uint64_t>(dim) << 56) | key;
}

void SparseTensor::init(
    std::string& name, std::vector<uint64_t>& shape, BaseKV* kv) {
  this->name  = name;
  this->shape = shape;
  this->kv    = kv;
}

void SparseTensor::Get(
    const uint64_t key, std::string& value, DIM_TYPE dim, unsigned tid) {
  auto _key = concatKeyAndDim(key, dim);
  kv->Get(_key, value, tid);
}

void SparseTensor::Put(const uint64_t key,
                       const std::string_view& value,
                       DIM_TYPE dim,
                       unsigned tid) {
  auto _key = concatKeyAndDim(key, dim);
  kv->Put(_key, value, tid);
}