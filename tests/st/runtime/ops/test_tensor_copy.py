# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical copy coverage, including SRAM reuse across InCore calls."""

import pypto.language as pl
import pytest
import torch


@pl.jit.incore
def stage(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.tensor.copy(dst, src, [0, 0], [0, 0], [4, 600])
    return dst


@pl.jit.incore
def restore(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.tensor.copy(
        dst,
        src,
        [0, 0],
        [0, 0],
        [4, 600],
        source_memory=pl.Mem.SRAM,
        target_memory=pl.Mem.DDR,
    )
    return dst


@pl.jit
def roundtrip(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    scratch = pl.create_tensor([4, 600], dtype=pl.FP32)
    scratch = stage(src, scratch)
    return restore(scratch, dst)


@pl.jit.incore
def copy_region(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.InOut[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.tensor.copy(dst, src, [1, 8], [0, 16], [2, 577])
    return dst


@pl.jit
def region(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.InOut[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    return copy_region(src, dst)


def test_cross_incore_roundtrip(test_config):
    src = torch.arange(2400, dtype=torch.float32).reshape(4, 600)
    dst = torch.zeros_like(src)
    roundtrip(src, dst, config=test_config)
    torch.testing.assert_close(dst, src, rtol=0, atol=0)


def test_subregion_preserves_surrounding_data(test_config):
    src = torch.arange(2400, dtype=torch.float32).reshape(4, 600)
    dst = torch.full_like(src, -1)
    expected = dst.clone()
    expected[1:3, 8:585] = src[:2, 16:593]
    region(src, dst, config=test_config)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
