// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=1 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget | FileCheck %s --check-prefix=TILE
// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=1 wram-bytes=2048 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget -pim-explicit-dma -pim-lower-to-emitc | FileCheck %s --check-prefix=TAIL

// 三个维度都不是 2 的幂。`pim.target` 让 `verifyTensorSize` 跳过 2 的幂检查，
// 否则这条输入进不了分块搜索。容量够时分块取整维本身；容量不够时分块取
// 2 的幂，余数由降级侧的下标夹取处理，循环上界仍是真实维度。
// TILE-DAG: "pim.tile-m" = 40 : i64
// TILE-DAG: "pim.tile-n" = 88 : i64
// TILE-DAG: "pim.tile-k" = 48 : i64
// TAIL-DAG: "emitc.constant"() <{value = 40 : i32}>
// TAIL-DAG: "emitc.constant"() <{value = 88 : i32}>
// TAIL-DAG: "emitc.constant"() <{value = 48 : i32}>
// 夹取的上界是真实维度减一：越界下标被按到最后一个合法位置。
// TAIL: "emitc.constant"() <{value = 87 : i32}>
// TAIL: cmp gt
// TAIL: conditional
module attributes {pim.target = "pim:v1"} {
  tt.func public @odd_linear(%a_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>,
                             %o_ptr: !tt.ptr<f16>) {
    %arange_m = tt.make_range {end = 40 : i32, start = 0 : i32} : tensor<40xi32>
    %arange_k = tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32>
    %arange_n = tt.make_range {end = 88 : i32, start = 0 : i32} : tensor<88xi32>

    %m2 = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<40xi32> -> tensor<40x1xi32>
    %k2 = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<48xi32> -> tensor<1x48xi32>
    %m2b = tt.broadcast %m2 : tensor<40x1xi32> -> tensor<40x48xi32>
    %k2b = tt.broadcast %k2 : tensor<1x48xi32> -> tensor<40x48xi32>
    %ck = arith.constant dense<48> : tensor<40x48xi32>
    %m_scaled = arith.muli %m2b, %ck : tensor<40x48xi32>
    %a_off = arith.addi %m_scaled, %k2b : tensor<40x48xi32>
    %a_ptrs0 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<40x48x!tt.ptr<f16>>
    %a_ptrs = tt.addptr %a_ptrs0, %a_off : tensor<40x48x!tt.ptr<f16>>, tensor<40x48xi32>
    %a = tt.load %a_ptrs : tensor<40x48x!tt.ptr<f16>>

    %n2 = tt.expand_dims %arange_n {axis = 1 : i32} : tensor<88xi32> -> tensor<88x1xi32>
    %k2_ = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<48xi32> -> tensor<1x48xi32>
    %n2b = tt.broadcast %n2 : tensor<88x1xi32> -> tensor<88x48xi32>
    %k2b_ = tt.broadcast %k2_ : tensor<1x48xi32> -> tensor<88x48xi32>
    %ck2 = arith.constant dense<48> : tensor<88x48xi32>
    %n_scaled = arith.muli %n2b, %ck2 : tensor<88x48xi32>
    %w_off = arith.addi %n_scaled, %k2b_ : tensor<88x48xi32>
    %w_ptrs0 = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<88x48x!tt.ptr<f16>>
    %w_ptrs = tt.addptr %w_ptrs0, %w_off : tensor<88x48x!tt.ptr<f16>>, tensor<88x48xi32>
    %w = tt.load %w_ptrs : tensor<88x48x!tt.ptr<f16>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<88x48xf16> -> tensor<48x88xf16>

    %acc0 = arith.constant dense<0.000000e+00> : tensor<40x88xf32>
    %d0 = tt.dot %a, %wt, %acc0 : tensor<40x48xf16> * tensor<48x88xf16> -> tensor<40x88xf32>
    %d = arith.truncf %d0 : tensor<40x88xf32> to tensor<40x88xf16>

    %m2o = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<40xi32> -> tensor<40x1xi32>
    %n2o = tt.expand_dims %arange_n {axis = 0 : i32} : tensor<88xi32> -> tensor<1x88xi32>
    %m2ob = tt.broadcast %m2o : tensor<40x1xi32> -> tensor<40x88xi32>
    %n2ob = tt.broadcast %n2o : tensor<1x88xi32> -> tensor<40x88xi32>
    %cn = arith.constant dense<88> : tensor<40x88xi32>
    %m_scaled_o = arith.muli %m2ob, %cn : tensor<40x88xi32>
    %o_off = arith.addi %m_scaled_o, %n2ob : tensor<40x88xi32>
    %o_ptrs0 = tt.splat %o_ptr : !tt.ptr<f16> -> tensor<40x88x!tt.ptr<f16>>
    %o_ptrs = tt.addptr %o_ptrs0, %o_off : tensor<40x88x!tt.ptr<f16>>, tensor<40x88xi32>
    tt.store %o_ptrs, %d : tensor<40x88x!tt.ptr<f16>>
    tt.return
  }
}
