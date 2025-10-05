import sgl.device.print;
struct FlashAttnInput
{
    StructuredBuffer<float> query;
    StructuredBuffer<float> key;
    StructuredBuffer<float> value;
    StructuredBuffer<bool> mask;
};

struct FlashAttnIntermediate
{
    RWStructuredBuffer<float> log_sum_exp;
    RWStructuredBuffer<float> maximum;
};

struct FlashAttnOutput
{
    RWStructuredBuffer<float> output;
};

ParameterBlock<FlashAttnInput> input;
ParameterBlock<FlashAttnIntermediate> intermediate;
ParameterBlock<FlashAttnOutput> output;

// Constants - these should be passed as parameters
#ifndef kBlockSizeKv
#define kBlockSizeKv 32
#endif

#ifndef kBlockSizeQ
#define kBlockSizeQ 32
#endif

#ifndef kBatchSize
#define kBatchSize 2
#endif

#ifndef kHeadSize
#define kHeadSize 4
#endif

#ifndef kSequenceLength
#define kSequenceLength 64
#endif

#ifndef kQueryDim
#define kQueryDim 32
#endif

#ifndef kValueDim
#define kValueDim kQueryDim
#endif

#ifndef kKeyDim
#define kKeyDim kQueryDim
#endif

#define kInf 1e9f
#define kTileCountKv ((kSequenceLength + kBlockSizeKv - 1) / kBlockSizeKv)
#define kTileCountQ ((kSequenceLength + kBlockSizeQ - 1) / kBlockSizeQ)

#define kSharedMemOffsetQuery 0
#define kSharedMemOffsetKey (kSharedMemOffsetQuery + kBlockSizeQ * kQueryDim)
#define kSharedMemOffsetValue (kSharedMemOffsetKey + kBlockSizeKv * kKeyDim)
#define kSharedMemOffsetScore (kSharedMemOffsetValue + kBlockSizeKv * kValueDim)
#define kSharedMemSize (kSharedMemOffsetScore + kBlockSizeKv * kBlockSizeQ)

groupshared float shared_mem[kSharedMemSize];

int32_t query_idx(int32_t idx)
{
    return idx + kSharedMemOffsetQuery;
}

int32_t key_idx(int32_t idx)
{
    return idx + kSharedMemOffsetKey;
}

int32_t value_idx(int32_t idx)
{
    return idx + kSharedMemOffsetValue;
}

int32_t score_idx(int32_t idx)
{
    return idx + kSharedMemOffsetScore;
}

[shader("compute")]
[numthreads(kBlockSizeKv, 1, 1)]
void flash_attn_kernel(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    let seq_local_idx = gtid.x; // sequence local index
    let batch_idx = gid.y;      // batch index
    let head_idx = gid.x;       // head index

    let qkv_offset = (batch_idx * kHeadSize * kSequenceLength * kQueryDim) + (head_idx * kSequenceLength * kQueryDim);
    let lm_offset = (batch_idx * kHeadSize * kSequenceLength) + (head_idx * kSequenceLength);

    let softmax_scale = 1.0f / sqrt(kQueryDim);

    for (int tile_kv_i = 0; tile_kv_i < kTileCountKv; ++tile_kv_i)
    {
        // Load key and value to shared memory then sync
        for (int i = 0; i < kKeyDim; ++i)
        {
            let idx = (seq_local_idx * kKeyDim) + i;
            shared_mem[key_idx(idx)] = input.key[qkv_offset + (kBlockSizeKv * kKeyDim * tile_kv_i) + idx];
        }
        for (int i = 0; i < kValueDim; ++i)
        {
            let idx = (seq_local_idx * kValueDim) + i;
            shared_mem[value_idx(idx)] = input.value[qkv_offset + (kBlockSizeKv * kValueDim * tile_kv_i) + idx];
        }
        DeviceMemoryBarrierWithGroupSync();

        for (int tile_q_i = 0; tile_q_i < kTileCountQ; ++tile_q_i)
        {
            // load query to shared memory, maximum and log-sum-exp to registers
            for (int i = 0; i < kQueryDim; ++i)
            {
                let idx = (seq_local_idx * kQueryDim) + i;
                shared_mem[query_idx(idx)] = input.query[qkv_offset + (kBlockSizeQ * kQueryDim * tile_q_i) + idx];
            }
            let row_maximum_prev = intermediate.maximum[lm_offset + (kBlockSizeQ * tile_q_i) + seq_local_idx];
            let row_log_sum_exp_prev = intermediate.log_sum_exp[lm_offset + (kBlockSizeQ * tile_q_i) + seq_local_idx];

            // compute new maximum
            var row_maximum = -kInf;
            for (int block_kv_i = 0; block_kv_i < kBlockSizeKv; ++block_kv_i)
            {
                var sum = 0.0f;
                for (int i = 0; i < kQueryDim; ++i)
                {
                    sum += shared_mem[query_idx((seq_local_idx * kQueryDim) + i)] * shared_mem[key_idx((block_kv_i * kKeyDim) + i)];
                }
                sum *= softmax_scale;
                shared_mem[score_idx((kBlockSizeKv * seq_local_idx) + block_kv_i)] = sum;
                row_maximum = max(sum, row_maximum);
            }

            // compute log-sum-exp
            var row_log_sum_exp = 0.0f;
            for (int block_kv_i = 0; block_kv_i < kBlockSizeKv; ++block_kv_i)
            {
                let idx = (kBlockSizeKv * seq_local_idx) + block_kv_i;
                shared_mem[score_idx(idx)] = exp(shared_mem[score_idx(idx)] - row_maximum);
                row_log_sum_exp += shared_mem[score_idx(idx)];
            }

            // update maximum and log-sum-exp
            let row_maximum_new = max(row_maximum_prev, row_maximum);
            let row_log_sum_exp_new = (exp(row_maximum_prev - row_maximum_new) * row_log_sum_exp_prev) + (exp(row_maximum - row_maximum_new) * row_log_sum_exp);

            // write results to output
            for (int value_i = 0; value_i < kValueDim; ++value_i)
            {
                var pv = 0.0f;
                for (int block_kv_i = 0; block_kv_i < kBlockSizeKv; ++block_kv_i)
                {
                    pv += shared_mem[score_idx((kBlockSizeKv * seq_local_idx) + block_kv_i)] * shared_mem[value_idx((block_kv_i * kValueDim) + value_i)];
                }
                let output_idx = qkv_offset + (kBlockSizeQ * kValueDim * tile_q_i) + (seq_local_idx * kValueDim) + value_i;
                output.output[output_idx] = (1.0f / row_log_sum_exp_new) * ((row_log_sum_exp_prev * exp(row_maximum_prev - row_maximum_new) * output.output[output_idx]) + (exp(row_maximum - row_maximum_new) * pv));
            }
            intermediate.maximum[lm_offset + (kBlockSizeQ * tile_q_i) + seq_local_idx] = row_maximum_new;
            intermediate.log_sum_exp[lm_offset + (kBlockSizeQ * tile_q_i) + seq_local_idx] = row_log_sum_exp_new;
        }
        DeviceMemoryBarrierWithGroupSync();
    }
}
