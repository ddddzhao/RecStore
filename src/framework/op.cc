#include "framework/op.h"
#include "grpc_ps/grpc_ps_client.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace recstore {

namespace {

GRPCParameterClient& GetGRPCClientInstance() {
    const std::string PS_HOST = "localhost";
    const int PS_PORT = 50051;
    const int SHARD_ID = 0;
    static std::unique_ptr<GRPCParameterClient> grpc_client =
        std::make_unique<GRPCParameterClient>(PS_HOST, PS_PORT, SHARD_ID);
    return *grpc_client;
}

} // anonymous namespace

}

namespace recstore {

void validate_keys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " + base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument(
        "Keys tensor must be 1-dimensional, but has " + std::to_string(keys.dim()) + " dimensions.");
  }
}

void validate_embeddings(const base::RecTensor& embeddings, const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " + base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " + std::to_string(embeddings.dim()) + " dimensions.");
  }
  if (embeddings.shape(1) != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument(name + " tensor has embedding dimension " + std::to_string(embeddings.shape(1)) +
                                ", but expected " + std::to_string(base::EMBEDDING_DIMENSION_D));
  }
}

void EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but values has length " +
                                std::to_string(values.shape(0)));
  }

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  float* values_data        = values.data_as<float>();
    auto keys_ptr = keys.data_as<const uint64_t>();
    base::ConstArray<uint64_t> key_array_view(keys_ptr, L);
    float* values_ptr = values.data_as<float>();

    bool success = GetGRPCClientInstance().GetParameter(key_array_view, values_ptr, /* perf= */ true);

    if (!success) {
        throw std::runtime_error("EmbRead: gRPC GetParameter failed.");
    }
    // std::cout << "[EmbRead] Read operation complete." << std::endl;
}


void EmbUpdate(const base::RecTensor& keys, const base::RecTensor& grads) {
  validate_keys(keys);
  validate_embeddings(grads, "Grads");

  const int64_t L = keys.shape(0);
  if (grads.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but grads has length " +
                                std::to_string(grads.shape(0)));
  }

    auto keys_ptr = keys.data_as<const uint64_t>();
    std::vector<uint64_t> keys_vec(keys_ptr, keys_ptr + L);

    auto grads_ptr = grads.data_as<const float>();
    int64_t emb_dim = grads.shape(1);
    std::vector<std::vector<float>> grads_vec;
    grads_vec.reserve(L);
    for (int64_t i = 0; i < L; ++i) {
        const float* row_start = grads_ptr + i * emb_dim;
        grads_vec.emplace_back(row_start, row_start + emb_dim);
    }

    bool success = GetGRPCClientInstance().PutParameter(keys_vec, grads_vec);

    if (!success) {
        throw std::runtime_error("EmbUpdate: gRPC PutParameter failed.");
    }
    // std::cout << "[EmbUpdate] Update operation complete." << std::endl;
}

namespace testing {

void ClearEmbeddingTableForTesting() {
    bool success = GetGRPCClientInstance().ClearPS();
    if (!success) {
        throw std::runtime_error("Failed to clear remote Parameter Server state during testing.");
    }
}

} // namespace testing

} // namespace recstore
