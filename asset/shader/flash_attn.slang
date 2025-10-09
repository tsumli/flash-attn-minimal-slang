import sgl.device.print;
struct FlashAttnInput
{
    // [B, H, Nq, Dq]
    StructuredBuffer<float> query;

    // [B, H, Nk, Dk]
    StructuredBuffer<float> key;

    // [B, H, Nk, Dv]
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
#ifndef kBlockSize
#define kBlockSize 32
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
#define kTileCount ((kSequenceLength + kBlockSize - 1) / kBlockSize)

#define kSharedMemOffsetQuery 0
#define kSharedMemOffsetKey (kSharedMemOffsetQuery + kBlockSize * kQueryDim)
#define kSharedMemOffsetValue (kSharedMemOffsetKey + kBlockSize * kKeyDim)
#define kSharedMemOffsetScore (kSharedMemOffsetValue + kBlockSize * kValueDim)
#define kSharedMemSize (kSharedMemOffsetScore + kBlockSize * kBlockSize)

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

void load_key_value_to_shared_mem(int32_t q_row_in_tile, int32_t kq_global_row, int32_t k_base, int32_t v_base)
{
    // load key to shared memory
    for (int i = 0; i < kKeyDim; ++i)
    {
        let idx = q_row_in_tile * kKeyDim + i;
        shared_mem[key_idx(idx)] = (kq_global_row < kSequenceLength) ? input.key[k_base + kq_global_row * kKeyDim + i] : 0.0f;
    }

    // load value to shared memory
    for (int i = 0; i < kValueDim; ++i)
    {
        let idx = q_row_in_tile * kValueDim + i;
        shared_mem[value_idx(idx)] = (kq_global_row < kSequenceLength) ? input.value[v_base + kq_global_row * kValueDim + i] : 0.0f;
    }
}

void load_query_to_shared_mem(int32_t q_row_in_tile, int32_t q_global_row, int32_t q_base)
{
    for (int i = 0; i < kQueryDim; ++i)
    {
        let idx = q_row_in_tile * kQueryDim + i;
        shared_mem[query_idx(idx)] = (q_global_row < kSequenceLength) ? input.query[q_base + q_global_row * kQueryDim + i] : 0.0f;
    }
}

// Compute attention scores for a query row against all keys in the current tile
float compute_attention_scores(int32_t q_row_in_tile, int32_t tile_kv_i, float softmax_scale)
{
    float row_maximum = -kInf;

    for (int block_kv_i = 0; block_kv_i < kBlockSize; ++block_kv_i)
    {
        int k_global_col = tile_kv_i * kBlockSize + block_kv_i;
        float sum = 0.0f;

        if (k_global_col < kSequenceLength)
        {
            for (int i = 0; i < kQueryDim; ++i)
            {
                sum += shared_mem[query_idx(q_row_in_tile * kQueryDim + i)] * shared_mem[key_idx(block_kv_i * kKeyDim + i)];
            }
            sum *= softmax_scale;
        }
        else
        {
            sum = -kInf;
        }

        shared_mem[score_idx(kBlockSize * q_row_in_tile + block_kv_i)] = sum;
        row_maximum = max(sum, row_maximum);
    }

    return row_maximum;
}

// Apply softmax to attention scores and compute log sum exp
float apply_softmax_and_compute_lse(int32_t q_row_in_tile, float row_maximum)
{
    float row_log_sum_exp = 0.0f;

    if (row_maximum == -kInf)
    {
        for (int block_kv_i = 0; block_kv_i < kBlockSize; ++block_kv_i)
        {
            shared_mem[score_idx(kBlockSize * q_row_in_tile + block_kv_i)] = 0.0f;
        }
    }
    else
    {
        for (int block_kv_i = 0; block_kv_i < kBlockSize; ++block_kv_i)
        {
            int idx = kBlockSize * q_row_in_tile + block_kv_i;
            float e = exp(shared_mem[score_idx(idx)] - row_maximum);
            shared_mem[score_idx(idx)] = e;
            row_log_sum_exp += e;
        }
    }

    return row_log_sum_exp;
}

// Update running maximum and log sum exp with online softmax algorithm
void update_running_stats(float row_maximum_prev, float row_log_sum_exp_prev,
                          float row_maximum, float row_log_sum_exp,
                          out float row_maximum_new, out float row_lse_new,
                          out float coeff_prev, out float coeff_blk)
{
    row_maximum_new = max(row_maximum_prev, row_maximum);
    coeff_prev = exp(row_maximum_prev - row_maximum_new) * row_log_sum_exp_prev;
    coeff_blk = (row_maximum == -kInf) ? 0.0f : exp(row_maximum - row_maximum_new);
    row_lse_new = coeff_prev + coeff_blk * row_log_sum_exp;
}

// Compute weighted value aggregation and update output
void compute_and_update_output(int32_t q_row_in_tile, int32_t q_global_row, int32_t q_base,
                               float coeff_prev, float coeff_blk, float row_lse_new,
                               float row_log_sum_exp_prev)
{
    int out_base = q_base + q_global_row * kValueDim;

    if (row_lse_new > 0.0f)
    {
        for (int v = 0; v < kValueDim; ++v)
        {
            float pv = 0.0f;
            for (int block_kv_i = 0; block_kv_i < kBlockSize; ++block_kv_i)
            {
                pv += shared_mem[score_idx(kBlockSize * q_row_in_tile + block_kv_i)] * shared_mem[value_idx(block_kv_i * kValueDim + v)];
            }
            float out_prev_term = (row_log_sum_exp_prev > 0.0f) ? (coeff_prev * output.output[out_base + v]) : 0.0f;
            output.output[out_base + v] = (out_prev_term + coeff_blk * pv) / row_lse_new;
        }
    }
}

// Process a single query tile against the current key-value tile
void process_query_tile(int32_t q_row_in_tile, int32_t tile_q_i, int32_t tile_kv_i,
                        int32_t q_base, int32_t lm_base, float softmax_scale)
{
    let q_global_row = tile_q_i * kBlockSize + q_row_in_tile;
    load_query_to_shared_mem(q_row_in_tile, q_global_row, q_base);

    if (q_global_row >= kSequenceLength)
    {
        return;
    }

    float row_maximum_prev = intermediate.maximum[lm_base + q_global_row];
    float row_log_sum_exp_prev = intermediate.log_sum_exp[lm_base + q_global_row];

    // Compute attention scores
    float row_maximum = compute_attention_scores(q_row_in_tile, tile_kv_i, softmax_scale);

    // Apply softmax and compute log sum exp
    float row_log_sum_exp = apply_softmax_and_compute_lse(q_row_in_tile, row_maximum);

    // Update running statistics
    float row_maximum_new, row_lse_new, coeff_prev, coeff_blk;
    update_running_stats(row_maximum_prev, row_log_sum_exp_prev, row_maximum, row_log_sum_exp,
                         row_maximum_new, row_lse_new, coeff_prev, coeff_blk);

    // Compute and update output
    compute_and_update_output(q_row_in_tile, q_global_row, q_base, coeff_prev, coeff_blk,
                              row_lse_new, row_log_sum_exp_prev);

    // Store updated statistics
    intermediate.maximum[lm_base + q_global_row] = row_maximum_new;
    intermediate.log_sum_exp[lm_base + q_global_row] = row_lse_new;
}

[shader("compute")]
[numthreads(kBlockSize, 1, 1)]
void flash_attn_kernel(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    let q_row_in_tile = gtid.x;
    let batch_idx = gid.y;
    let head_idx = gid.x;

    let q_base = (batch_idx * kHeadSize * kSequenceLength * kQueryDim) + (head_idx * kSequenceLength * kQueryDim);
    let k_base = (batch_idx * kHeadSize * kSequenceLength * kKeyDim) + (head_idx * kSequenceLength * kKeyDim);
    let v_base = (batch_idx * kHeadSize * kSequenceLength * kValueDim) + (head_idx * kSequenceLength * kValueDim);

    let lm_base = (batch_idx * kHeadSize * kSequenceLength) + (head_idx * kSequenceLength);
    let softmax_scale = 1.0f / sqrt((float)kQueryDim);

    for (int tile_kv_i = 0; tile_kv_i < kTileCount; ++tile_kv_i)
    {
        let kq_global_row = tile_kv_i * kBlockSize + q_row_in_tile;
        load_key_value_to_shared_mem(q_row_in_tile, kq_global_row, k_base, v_base);
        DeviceMemoryBarrierWithGroupSync();

        for (int tile_q_i = 0; tile_q_i < kTileCount; ++tile_q_i)
        {
            process_query_tile(q_row_in_tile, tile_q_i, tile_kv_i, q_base, lm_base, softmax_scale);
        }
        DeviceMemoryBarrierWithGroupSync();
    }
}
