add_library(sparse_tensor
	sparse_tensor.cpp
)

target_link_libraries(sparse_tensor PUBLIC
    kv_engine
)
