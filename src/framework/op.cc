#include "framework/op.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <numeric>
#include <thread>

// Assuming InitStrategyType is defined in base/tensor.h
#include "base/tensor.h"
#include "op.h"

#ifdef USE_FAKE_KVCLIENT
namespace recstore {
FakeKVClientOp::FakeKVClientOp() {}
void FakeKVClientOp::EmbInit(const base::RecTensor& keys, const base::RecTensor& init_values) {
    EmbWrite(keys, init_values);
}
void FakeKVClientOp::EmbInit(const base::RecTensor& keys, const InitStrategy& strategy) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (embedding_dim_ == -1) {
        throw std::runtime_error("FakeKVClientOp Error: Embedding dimension has not been set.");
    }
    const uint64_t* key_data = keys.data_as<uint64_t>();
    const int64_t num_keys = keys.shape(0);
    const int64_t emb_dim = this->embedding_dim_;
    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        store_[key] = std::vector<float>(emb_dim, 0.0f);
    }
}
void FakeKVClientOp::EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = values.shape(1);
    if (embedding_dim_ != -1 && embedding_dim_ != emb_dim) {
        throw std::runtime_error("FakeKVClientOp Error: Inconsistent embedding dimension for read.");
    }
    const uint64_t* key_data = keys.data_as<uint64_t>();
    float* value_data = values.data_as<float>();
    const int64_t num_keys = keys.shape(0);
    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        auto it = store_.find(key);
        if (it == store_.end()) {
            std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
        } else {
            std::copy(it->second.begin(), it->second.end(), value_data + i * emb_dim);
        }
    }
}
void FakeKVClientOp::EmbWrite(const base::RecTensor& keys, const base::RecTensor& values) {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = values.shape(1);
    if (embedding_dim_ == -1) {
        embedding_dim_ = emb_dim;
    } else if (embedding_dim_ != emb_dim) {
        throw std::runtime_error("FakeKVClientOp Error: Inconsistent embedding dimension for write.");
    }
    const uint64_t* key_data = keys.data_as<uint64_t>();
    const float* value_data = values.data_as<float>();
    const int64_t num_keys = keys.shape(0);
    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        const float* start = value_data + i * emb_dim;
        const float* end = start + emb_dim;
        store_[key].assign(start, end);
    }
}
void FakeKVClientOp::EmbUpdate(const base::RecTensor& keys, const base::RecTensor& grads) {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = grads.shape(1);
    if (embedding_dim_ == -1) {
        embedding_dim_ = emb_dim;
    } else if (embedding_dim_ != emb_dim) {
        throw std::runtime_error("FakeKVClientOp Error: Inconsistent embedding dimension for update.");
    }
    const uint64_t* key_data = keys.data_as<uint64_t>();
    const float* grad_data = grads.data_as<float>();
    const int64_t num_keys = keys.shape(0);
    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        auto it = store_.find(key);
        if (it != store_.end()) {
            for (int64_t j = 0; j < emb_dim; ++j) {
                it->second[j] -= learning_rate_ * grad_data[i * emb_dim + j];
            }
        }
    }
}
bool FakeKVClientOp::EmbExists(const base::RecTensor& keys) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::EmbDelete(const base::RecTensor& keys) { throw std::runtime_error("Not impl"); }
uint64_t FakeKVClientOp::EmbPrefetch(const base::RecTensor& keys) { throw std::runtime_error("Not impl"); }
bool FakeKVClientOp::IsPrefetchDone(uint64_t prefetch_id) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::WaitForPrefetch(uint64_t prefetch_id) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::GetPretchResult(uint64_t prefetch_id, std::vector<std::vector<float>>* values) { throw std::runtime_error("Not impl"); }
uint64_t FakeKVClientOp::EmbWriteAsync(const base::RecTensor& keys, const base::RecTensor& values) { throw std::runtime_error("Not impl"); }
bool FakeKVClientOp::IsWriteDone(uint64_t write_id) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::WaitForWrite(uint64_t write_id) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::SaveToFile(const std::string& path) { throw std::runtime_error("Not impl"); }
void FakeKVClientOp::LoadFromFile(const std::string& path) { throw std::runtime_error("Not impl"); }

std::shared_ptr<CommonOp> GetKVClientOp() {
    static std::shared_ptr<CommonOp> instance;
    static std::once_flag once_flag;
    std::call_once(once_flag, []() {
        instance = std::make_shared<FakeKVClientOp>();
    });
    return instance;
}
} // namespace recstore
#else
namespace recstore {
BasePSClient* KVClientOp::ps_client_ = nullptr;

void KVClientOp::EmbRead(const RecTensor& keys, RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  ConstArray<uint64_t> keys_array(keys_data, L);
  float* values_data = values.data_as<float>();
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  bool success = ps_client_->GetParameter(keys_array, values_data);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

void KVClientOp::EmbWrite(const RecTensor& keys,
                          const RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const int64_t D = values.shape(1);

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  ConstArray<uint64_t> keys_array(keys_data, L);
  const float* values_data = values.data_as<float>();
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  const std::vector<std::vector<float>> values_vector(L, std::vector<float>(D));
  for (int64_t i = 0; i < L; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      values_vector[i][j] = values_data[i * D + j];
    }
  }
  bool success = ps_client_->PutParameter(keys_array, values_vector);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

uint64_t KVClientOp::EmbPrefetch(const base::RecTensor& keys) {
  auto prefetch_id = ps_client_->PrefetchParameter(keys);

  return prefetch_id;
}

bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
  return ps_client_->IsPrefetchDone(prefetch_id);
}

void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
  while (!IsPrefetchDone(prefetch_id)) {
    // do nothing
  }
}

void KVClientOp::GetPretchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) override {
  ps_client_->GetPrefetchResult(prefetch_id, values);
}

std::shared_ptr<CommonOp> GetKVClientOp() {
  static std::shared_ptr<CommonOp> instance;
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    instance = std::make_shared<KVClientOp>();
  });
  return instance;
}

} // namespace recstore
#endif