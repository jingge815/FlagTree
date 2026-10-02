// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=32 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget='full-m=8 full-n=8 full-k=8' | FileCheck %s --check-prefix=OK

// FlagGems linear_kernel 的真实形态：按 2 维 launch grid 划分 M/N，一个 program
// 只算一块。可见分块 8x8x4 占 384 字节，远超 wram-bytes=32，但**不能**改写——
// 重建出来的是「一个 program 算完整张输出」，而启动网格不变，语义就变了。
// 所以这里只按实测分块回写属性，内核原样保留。

module {
  tt.func public @flaggems_linear(%x_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>,
                                  %o_ptr: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %zero_x = arith.constant dense<false> : tensor<8x4xi1>
    %zero_w = arith.constant dense<false> : tensor<4x8xi1>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %step_x = arith.constant dense<4> : tensor<8x4xi32>
    %step_w = arith.constant dense<4> : tensor<4x8xi32>
    %x_off0 = arith.constant dense<0> : tensor<8x4xi32>
    %w_off0 = arith.constant dense<0> : tensor<4x8xi32>

    %m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %n = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>

    // grid 划分：program id 参与指针计算，这是「一个 program 只算一块」的标志。
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %m_base = arith.muli %pid_m, %c2 : i32
    %n_base = arith.muli %pid_n, %c2 : i32
    %pid_x = arith.addi %m_base, %n_base : i32
    %pid_x32 = arith.muli %pid_x, %c1 : i32

    %x_off = tt.splat %pid_x32 : i32 -> tensor<8x4xi32>
    %x_base = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<8x4x!tt.ptr<f16>>
    %x_ptrs0 = tt.addptr %x_base, %x_off : tensor<8x4x!tt.ptr<f16>>, tensor<8x4xi32>
    %w_off = tt.splat %pid_x32 : i32 -> tensor<4x8xi32>
    %w_base = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<4x8x!tt.ptr<f16>>
    %w_ptrs0 = tt.addptr %w_base, %w_off : tensor<4x8x!tt.ptr<f16>>, tensor<4x8xi32>

    %loop:3 = scf.for %i = %c0 to %c2 step %c1 iter_args(%xp = %x_ptrs0, %wp = %w_ptrs0, %acc = %acc0) -> (tensor<8x4x!tt.ptr<f16>>, tensor<4x8x!tt.ptr<f16>>, tensor<8x8xf32>) : i32 {
      %a = tt.load %xp, %zero_x : tensor<8x4x!tt.ptr<f16>>
      // 权重按 K 主序加载，得到的就是 [K, N]，没有 tt.trans。
      %b = tt.load %wp, %zero_w : tensor<4x8x!tt.ptr<f16>>
      %d = tt.dot %a, %b, %acc : tensor<8x4xf16> * tensor<4x8xf16> -> tensor<8x8xf32>
      %xp2 = tt.addptr %xp, %step_x : tensor<8x4x!tt.ptr<f16>>, tensor<8x4xi32>
      %wp2 = tt.addptr %wp, %step_w : tensor<4x8x!tt.ptr<f16>>, tensor<4x8xi32>
      scf.yield %xp2, %wp2, %d : tensor<8x4x!tt.ptr<f16>>, tensor<4x8x!tt.ptr<f16>>, tensor<8x8xf32>
    }
    %out = arith.truncf %loop#2 : tensor<8x8xf32> to tensor<8x8xf16>

    %m3 = tt.expand_dims %m {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %n3 = tt.expand_dims %n {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %om = tt.broadcast %m3 : tensor<8x1xi32> -> tensor<8x8xi32>
    %on = tt.broadcast %n3 : tensor<1x8xi32> -> tensor<8x8xi32>
    %co = arith.constant dense<8> : tensor<8x8xi32>
    %o_scaled = arith.muli %om, %co : tensor<8x8xi32>
    %o_off = arith.addi %o_scaled, %on : tensor<8x8xi32>
    %o_base = tt.splat %o_ptr : !tt.ptr<f16> -> tensor<8x8x!tt.ptr<f16>>
    %o_ptrs = tt.addptr %o_base, %o_off : tensor<8x8x!tt.ptr<f16>>, tensor<8x8xi32>
    tt.store %o_ptrs, %out : tensor<8x8x!tt.ptr<f16>>
    tt.return
  }
}

// 分块属性回写的是实测分块（8x8x4），超预算这件事由
// pim.tile-wram-bytes > pim.wram-bytes 如实体现。
// 模块属性按名字排序打印，故顺序是 k、m、n。
// OK: "pim.tile-k" = 4 : i64
// OK: "pim.tile-m" = 8 : i64
// OK: "pim.tile-n" = 8 : i64
// OK: tt.func
// OK-NOT: tt.trans
// OK: tt.return
