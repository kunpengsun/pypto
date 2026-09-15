# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank all-to-all via ``pld.tensor.all_to_all`` intrinsic.

Validates the push-based composite all-to-all intrinsic produces the correct
rank-ordered personalized exchange.

The intrinsic accepts three arguments: ``input`` (Tensor [NR, SIZE]),
``target`` (DistributedTensor [NR, SIZE] window), ``signal`` (INT32 barrier),
and returns ``target`` in-place (window-as-result — same idiom as
``reduce_scatter`` / ``broadcast``).  The caller reads back from the
window into a plain output tensor for host-side verification.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices). Both use the same N-rank program body.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

STAGE_CHUNK = 4096


def _expected_all_to_all(inputs: torch.Tensor, size: int) -> torch.Tensor:
    """Golden matching simpler: output[rank, src, j] = src*1000 + rank*100 + j."""
    nranks = inputs.shape[0]
    src_idx = torch.arange(nranks, dtype=torch.float32).view(1, -1, 1)
    rank_idx = torch.arange(nranks, dtype=torch.float32).view(-1, 1, 1)
    j = torch.arange(size, dtype=torch.float32).view(1, 1, -1)
    return src_idx * 1000 + rank_idx * 100 + j


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r*1000 + d*100 + j (chunk for dest d)."""
    r = torch.arange(n_ranks, dtype=torch.float32).view(-1, 1, 1)
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(size, dtype=torch.float32).view(1, 1, -1)
    return r * 1000 + d * 100 + j


def _build_all_to_all_program(n_ranks: int, size: int):
    """Build an N-rank all-to-all program at call time using the intrinsic."""
    nr = n_ranks
    SIZE = size

    @pl.jit.incore
    def exchange_step(
        inp: pl.Tensor[[nr, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
        # Push-based all_to_all — intrinsic pushes chunks to peers and
        # returns data in-place (window-as-result).
        result = pld.tensor.all_to_all(inp, data, signal)
        # Read back from the window into out for host-side verification,
        # chunked through a fixed-width on-chip tile so a non-tile-aligned
        # or larger-than-UB SIZE never reserves an oversized or misaligned
        # tile (STAGE_CHUNK is a compile-time constant, always 32-byte
        # aligned; valid_shape masks the logical width actually read).
        for src in pl.range(nr):
            for col in pl.range(0, SIZE, STAGE_CHUNK):
                valid = pl.min(STAGE_CHUNK, SIZE - col)
                chunk = pl.load(result, [src, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
                pl.store(chunk, [src, col], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[nr, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
        return exchange_step(inp, out, data, signal)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


class TestL3TensorAllToAllIntrinsic:
    """L3 distributed runtime: N-rank all-to-all via ``pld.tensor.all_to_all``.

    Validates the push-based composite intrinsic produces a bit-identical
    rank-ordered personalized exchange.
    """

    @pytest.mark.parametrize("size", [17, 4097, 65537])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_intrinsic(self, test_config, device_ids, n_ranks, size):
        """Compile and run mesh all-to-all for P=2 or P=4; skip when devices are scarce.

        SIZE values cover the InCore staging-tile-cap fix: 17 (unaligned,
        exercises the floor-to-32-byte-aligned-width path — round DOWN, not
        up, since the stage tile can never exceed the transfer it slides
        through) and 4097 (one element past the 4096-element/16 KiB
        chunk-budget boundary for FP32). SIZE=1 is not covered here: below one
        32-byte alignment unit (8 FP32 elements) there is no stage width that
        both satisfies pto.alloc_tile's alignment rule and still fits within
        the transfer, a pre-existing gap this fix does not close. 65537 (the
        original bug repro size — a [1, 65537] FP32 stage would reserve 256
        KiB and overflow UB before this fix) requires running WITHOUT
        --forked: with --forked, this case hangs (blocked in
        futex_wait_queue, not an error) — this is a pre-existing `--forked`
        + sim fork()-thread-lock deadlock, unrelated to this fix. Without
        --forked, all three sizes pass cleanly for every composite in this
        file.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_all_to_all_program(n_ranks, size)
        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, n_ranks, size), dtype=torch.float32)
        compiled = program.compile(
            inputs,
            outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:n_ranks],
                    num_sub_workers=0,
                ),
            ),
        )
        compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))

        expected = _expected_all_to_all(inputs, size)
        assert torch.allclose(outputs, expected), (
            f"all-to-all intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
