# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank broadcast via ``pld.tensor.broadcast`` intrinsic.

Same on-board semantics as ``test_l3_broadcast.py`` — but the InCore body
calls the new composite intrinsic rather than hand-rolling notify/wait/remote_load.

Golden: every rank's output equals root's input.  Non-root inputs must not
appear in outputs.

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

ROOT_RANK = 0
STAGE_CHUNK = 4096
DYNAMIC_NRANKS = 2
BROADCAST_RUNTIME_SIZE = pl.dynamic("BROADCAST_RUNTIME_SIZE")


def _expected_broadcast(inputs: torch.Tensor, root: int = ROOT_RANK) -> torch.Tensor:
    """Root row replicated on every rank."""
    root_row = inputs[root, 0]
    return torch.stack([root_row] * inputs.shape[0]).unsqueeze(1)


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Distinct per-rank tensors so root-only broadcast is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_broadcast_program(n_ranks: int, size: int):
    """Build an N-rank broadcast program at call time using the intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks
    SIZE = size

    @pl.jit.incore
    def broadcast_step(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        # Phase 1: root only stages data, chunked through a fixed-width
        # on-chip tile so a non-tile-aligned or larger-than-UB SIZE never
        # reserves an oversized or misaligned tile (STAGE_CHUNK is a
        # compile-time constant, always 32-byte aligned; valid_shape masks
        # the logical width actually transferred).
        if my_rank == ROOT_RANK:
            for col in pl.range(0, SIZE, STAGE_CHUNK):
                valid = pl.min(STAGE_CHUNK, SIZE - col)
                local = pl.load(inp, [0, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
                pl.store(local, [0, col], data)

        # Phases 2-3: barrier + broadcast — one call.
        data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)

        # Stage-out: every rank reads root's data, same chunking as above.
        for col in pl.range(0, SIZE, STAGE_CHUNK):
            valid = pl.min(STAGE_CHUNK, SIZE - col)
            acc = pl.load(data, [0, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
            pl.store(acc, [0, col], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return broadcast_step(inp, out, data, signal, my_rank)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
        return outputs

    return host_orch


@pl.jit.incore
def _dynamic_broadcast_step(
    inp: pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[DYNAMIC_NRANKS, 1], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]:
    BROADCAST_RUNTIME_SIZE = pl.dynamic("BROADCAST_RUNTIME_SIZE")
    if my_rank == ROOT_RANK:
        for col in pl.range(0, BROADCAST_RUNTIME_SIZE, STAGE_CHUNK):
            valid = pl.min(STAGE_CHUNK, BROADCAST_RUNTIME_SIZE - col)
            local = pl.load(inp, [0, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
            pl.store(local, [0, col], data)
    data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
    for col in pl.range(0, BROADCAST_RUNTIME_SIZE, STAGE_CHUNK):
        valid = pl.min(STAGE_CHUNK, BROADCAST_RUNTIME_SIZE - col)
        acc = pl.load(data, [0, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
        pl.store(acc, [0, col], out)
    return out


@pl.jit
def _dynamic_broadcast_chip(
    inp: pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[DYNAMIC_NRANKS, 1], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, BROADCAST_RUNTIME_SIZE], pl.FP32]:
    return _dynamic_broadcast_step(inp, out, data, signal, my_rank)


@pl.jit.host
def _dynamic_broadcast_host(
    inputs: pl.Tensor[[DYNAMIC_NRANKS, 1, BROADCAST_RUNTIME_SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[DYNAMIC_NRANKS, 1, BROADCAST_RUNTIME_SIZE], pl.FP32]],
) -> pl.Tensor[[DYNAMIC_NRANKS, 1, BROADCAST_RUNTIME_SIZE], pl.FP32]:
    BROADCAST_RUNTIME_SIZE = pl.dynamic("BROADCAST_RUNTIME_SIZE")
    data_buf = pld.alloc_window_buffer(BROADCAST_RUNTIME_SIZE * pl.FP32.get_byte())
    signal_buf = pld.alloc_window_buffer(DYNAMIC_NRANKS * pl.INT32.get_byte())
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, BROADCAST_RUNTIME_SIZE], dtype=pl.FP32)
        sig = pld.window(signal_buf, [DYNAMIC_NRANKS, 1], dtype=pl.INT32)
        _dynamic_broadcast_chip(inputs[r], outputs[r], data, sig, r, device=r)
    return outputs


class TestL3TensorBroadcastIntrinsic:
    """L3 distributed runtime: N-rank broadcast via ``pld.tensor.broadcast``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_broadcast.py`` reference.
    """

    @pytest.mark.parametrize("size", [17, 4097, 65537])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_broadcast_intrinsic(self, test_config, device_ids, n_ranks, size):
        """Compile and run mesh broadcast for P=2 or P=4; skip when devices are scarce.

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
            pytest.skip(f"broadcast P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_broadcast_program(n_ranks, size)
        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)
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

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_broadcast_dynamic_target_shape_runtime(self, test_config, device_ids):
        """Compile and execute a target whose trailing extent is runtime-bound."""
        n_ranks = 2
        size = 17
        if len(device_ids) < n_ranks:
            pytest.skip(f"dynamic broadcast P=2 needs 2 devices, got {device_ids}")

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros_like(inputs)
        compiled = _dynamic_broadcast_host.compile(
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

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"dynamic broadcast mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
