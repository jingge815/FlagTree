// RUN: triton-opt %s -split-input-file -verify-diagnostics

// `dpusPerDevice` 的校验：跨 DPU 切分决策由图编译器写入，非法值必须在解析期
// 就被拒绝，而不是带着退化的决策往下走。
//
// 负例与正例分文件，是本目录的惯例（`*_negative.mlir`），也是必须的：
// `-verify-diagnostics` 让「报错」成为预期结果，而正例那几条 RUN 行不带这个
// 开关，两类混在一份文件里会让 `triton-opt` 非零退出 —— lit 默认
// `pipefail=True`，于是正例的 RUN 行整条判失败（实测如此）。
//
// RUN 行形态照 `operator_ops_negative.mlir` 那组：本文件的 expected-error 全部
// 命中，`triton-opt` 零退出，所以既不需要 `not` 也不需要 FileCheck。

// Zero DPUs on an axis is meaningless, and the verifier says so rather than
// letting a degenerate sharding decision through.
// expected-error @below {{dpusPerDevice must be positive in every dimension}}
#zero_dpus = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [0, 1], order = [1, 0]}>
module attributes {pim.target = "pim:v1"} {
  tt.func @zero_dpus_is_rejected(%t: tensor<2048x4096xf16, #zero_dpus>) {
    tt.return
  }
}

// -----

// `dpusPerDevice` must have the same rank as the rest of the encoding -- a
// rank-mismatched split would silently describe a different tensor.
// expected-error @below {{must all have the same rank}}
#short_dpus = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [2], order = [1, 0]}>
module attributes {pim.target = "pim:v1"} {
  tt.func @rank_mismatch_is_rejected(%t: tensor<2048x4096xf16, #short_dpus>) {
    tt.return
  }
}
