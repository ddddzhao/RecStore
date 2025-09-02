#!/bin/bash

# Usage:
# (with nohup)
# > training_output.4194304.$(date +%Y%m%d%H%M%S).log \
# 2>&1

export PROCESSED_DATASET_PATH="./processed_day_0_data"
export BATCH_SIZE=1024
export LEARNING_RATE=0.005
export EPOCHS=1

export DLRM_PATH="$(pwd)"
export SINGLE_DAY_TEST="${DLRM_PATH}/tests/dlrm_main_single_day.py"
source ${DLRM_PATH}/dlrm_venv/bin/activate

echo "=== TorchRec DLRM Single Day Training ==="
echo "Dataset path: $PROCESSED_DATASET_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo ""

# 检查处理后的数据是否存在
if [ ! -d "$PROCESSED_DATASET_PATH" ]; then
    echo "Error: Processed dataset not found at $PROCESSED_DATASET_PATH"
    echo "Please run the preprocessing script first:"
    echo "bash scripts/process_single_day.sh <raw_data_dir> $PROCESSED_DATASET_PATH"
    exit 1
fi

# 检查必要的文件
required_files=("day_0_dense.npy" "day_0_sparse.npy" "day_0_labels.npy")
for file in "${required_files[@]}"; do
    if [ ! -f "$PROCESSED_DATASET_PATH/$file" ]; then
        echo "Error: Required file $file not found in $PROCESSED_DATASET_PATH"
        exit 1
    fi
done

echo "✓ All required data files found"
echo ""

# 运行训练
echo "Starting training..."
torchrun --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost \
    --rdzv_id 54321 \
    --role trainer $SINGLE_DAY_TEST \
    --single_day_mode \
    --in_memory_binary_criteo_path $PROCESSED_DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --pin_memory \
    --mmap_mode \
    --embedding_dim 128 \
    --adagrad

echo ""
echo "Training completed!" 

source ~/.bashrc