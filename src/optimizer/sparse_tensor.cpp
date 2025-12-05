#include "sparse_tensor.h"

uint64_t SparseTensor::concatKeyAndDim(uint64_t key, TAG_TYPE tag) {
  key &= 0x00FFFFFFFFFFFFFF;
  return (static_cast<uint64_t>(tag) << 56) | key;
}

void SparseTensor::init(std::string& name,
                        TensorType type,
                        std::vector<uint64_t>& shape,
                        BaseKV* kv) {
  this->name  = name;
  this->type  = type;
  this->shape = shape;
  this->kv    = kv;
}

void SparseTensor::Get(
    const uint64_t key, std::string& value, TAG_TYPE tag, unsigned tid) {
  auto _key = concatKeyAndDim(key, tag);
  kv->Get(_key, value, tid);
}

void SparseTensor::Put(const uint64_t key,
                       const std::string_view& value,
                       TAG_TYPE tag,
                       unsigned tid) {
  auto _key = concatKeyAndDim(key, tag);
  kv->Put(_key, value, tid);
}