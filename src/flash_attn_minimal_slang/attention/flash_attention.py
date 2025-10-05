import slangpy as spy
from pathlib import Path
import numpy as np
import numpy.typing as npt

ASSET_ROOT = Path(__file__).parents[3] / "asset"


class FlashAttention:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_query_tokens: int,
        num_key_tokens: int,
        query_dim: int,
        key_dim: int,
        value_dim: int,
    ):
        # precondition
        assert num_query_tokens == num_key_tokens
        assert query_dim == key_dim

        super().__init__()

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.num_key_tokens = num_key_tokens
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.num_threads = (32, 1, 1)

        # create device and kernel
        self.device = spy.Device(
            enable_debug_layers=True,
            enable_print=True,
            compiler_options=spy.SlangCompilerOptions(
                {
                    "include_paths": [ASSET_ROOT / "shader"],
                    "defines": {
                        "kBlockSizeKv": str(32),
                        "kBlockSizeQ": str(32),
                        "kBatchSize": str(batch_size),
                        "kHeadSize": str(num_heads),
                        "kSequenceLength": str(num_query_tokens),
                        "kQueryDim": str(query_dim),
                        "kValueDim": str(value_dim),
                    },
                }
            ),
        )
        self.command_encoder = self.device.create_command_encoder()
        self.program = self.device.load_program(
            str(ASSET_ROOT / "shader" / "flash_attn.slang"), ["flash_attn_kernel"]
        )
        self.kernel = self.device.create_compute_kernel(self.program)

        # set shapes
        self.query_shape = (batch_size, num_heads, num_query_tokens, query_dim)
        self.key_shape = (batch_size, num_heads, num_key_tokens, key_dim)
        self.value_shape = (batch_size, num_heads, num_key_tokens, value_dim)
        self.mask_shape = (batch_size, 1, num_query_tokens, num_key_tokens)
        self.output_shape = (batch_size, num_heads, num_query_tokens, value_dim)
        self.log_sum_exp_shape = (batch_size, num_heads, num_query_tokens)
        self.maximum_shape = (batch_size, num_heads, num_query_tokens)

        # allocate buffers
        self.buffer_query: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.query_shape),
            resource_type_layout=self.kernel.reflection.input.query,
            usage=spy.BufferUsage.shader_resource,
            label="query",
        )
        self.buffer_key: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.key_shape),
            resource_type_layout=self.kernel.reflection.input.key,
            usage=spy.BufferUsage.shader_resource,
            label="key",
        )
        self.buffer_value: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.value_shape),
            resource_type_layout=self.kernel.reflection.input.value,
            usage=spy.BufferUsage.shader_resource,
            label="value",
        )
        self.buffer_mask: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.mask_shape),
            resource_type_layout=self.kernel.reflection.input.mask,
            usage=spy.BufferUsage.shader_resource,
            label="mask",
        )
        self.buffer_output: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.output_shape),
            resource_type_layout=self.kernel.reflection.output.output,
            usage=spy.BufferUsage.unordered_access,
            label="output",
        )
        self.buffer_log_sum_exp: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.log_sum_exp_shape),
            resource_type_layout=self.kernel.reflection.intermediate.log_sum_exp,
            usage=spy.BufferUsage.unordered_access,
            label="log_sum_exp",
        )
        self.buffer_maximum: spy.Buffer = self.device.create_buffer(
            element_count=np.prod(self.maximum_shape),
            resource_type_layout=self.kernel.reflection.intermediate.maximum,
            usage=spy.BufferUsage.unordered_access,
            label="maximum",
        )

    def __call__(
        self,
        q: npt.NDArray[np.float32],
        k: npt.NDArray[np.float32],
        v: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float32]:
        # precondition
        assert q.shape == self.query_shape
        assert k.shape == self.key_shape
        assert v.shape == self.value_shape
        assert mask.shape == self.mask_shape

        self.buffer_query.copy_from_numpy(q)
        self.buffer_key.copy_from_numpy(k)
        self.buffer_value.copy_from_numpy(v)
        self.buffer_mask.copy_from_numpy(mask)
        self.buffer_log_sum_exp.copy_from_numpy(
            np.zeros(np.prod(self.log_sum_exp_shape)).astype(np.float32)
        )
        self.buffer_maximum.copy_from_numpy(
            np.full(np.prod(self.maximum_shape), -np.inf).astype(np.float32)
        )

        group_count = (
            self.num_heads,
            self.batch_size,
            1,
        )
        self.kernel.dispatch(
            thread_count=(
                group_count[0] * self.num_threads[0],
                group_count[1] * self.num_threads[1],
                group_count[2] * self.num_threads[2],
            ),
            vars={
                "input": {
                    "query": self.buffer_query,
                    "key": self.buffer_key,
                    "value": self.buffer_value,
                    "mask": self.buffer_mask,
                },
                "intermediate": {
                    "log_sum_exp": self.buffer_log_sum_exp,
                    "maximum": self.buffer_maximum,
                },
                "output": {
                    "output": self.buffer_output,
                },
            },
            command_encoder=self.command_encoder,
        )

        self.device.submit_command_buffer(self.command_encoder.finish())
        self.device.flush_print()

        out = self.buffer_output.to_numpy().view(np.float32)
        out = out.reshape(self.output_shape)
        return out
