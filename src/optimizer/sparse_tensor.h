#include <string>
#include <vector>
#include "../storage/kv_engine/base_kv.h"

#define TAG_TYPE uint8_t

enum TensorType { PARAMETER = 0, MOMENT_1 = 1, MOMENT_2 = 2 };

class SparseTensor {
private:
  std::string name;
  TensorType type;
  std::vector<uint64_t> shape;
  BaseKV* kv;
  uint64_t concatKeyAndDim(uint64_t key, TAG_TYPE tag);

public:
  SparseTensor() = default;
  void init(std::string& name,
            TensorType type,
            std::vector<uint64_t>& shape,
            BaseKV* kv);
  void Get(const uint64_t key, std::string& value, TAG_TYPE tag, unsigned tid);
  void Put(const uint64_t key,
           const std::string_view& value,
           TAG_TYPE tag,
           unsigned tid);
};