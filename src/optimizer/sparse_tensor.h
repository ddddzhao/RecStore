#include <string>
#include <vector>
#include "../storage/kv_engine/base_kv.h"

#define DIM_TYPE uint8_t

class SparseTensor {
private:
  std::string name;
  std::vector<uint64_t> shape;
  BaseKV* kv;
  uint64_t concatKeyAndDim(uint64_t key, DIM_TYPE dim);

public:
  SparseTensor() = default;
  void init(std::string& name, std::vector<uint64_t>& shape, BaseKV* kv);
  void Get(const uint64_t key, std::string& value, DIM_TYPE dim, unsigned tid);
  void Put(const uint64_t key,
           const std::string_view& value,
           DIM_TYPE dim,
           unsigned tid);
};