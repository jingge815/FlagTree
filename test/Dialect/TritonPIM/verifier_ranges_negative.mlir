// RUN: triton-opt %s -split-input-file -verify-diagnostics

// 取值域缺口：写进 IR 的数不在硬件可表达的集合里时必须当场拒绝，
// 否则那个值会一路传到 GML，两侧都不报错。

module {
  tt.func @activation_mode_out_of_range(%x: tensor<4x8xf16>) {
    // expected-error @below {{activationMode is 0, 1 or 2}}
    %y = pim.lut %x {kind = #pim.activation<silu>, activationMode = 7 : i64}
       : tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

module {
  tt.func @spc_axis_out_of_range(%x: tensor<4x8xf16>, %b: tensor<8xf32>,
                                 %s: tensor<8xf16>) {
    // expected-error @below {{spcAxis}}
    %y = pim.fpsu_scale %x, %b, %s
        {shift = 0 : i8, spec = #pim.fpsu_spec<mode = fixed_point, spcAxis = 5>}
       : tensor<4x8xf16>, tensor<8xf32>, tensor<8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// 起始地址按字节计，必须是元素宽度的整数倍：f32 是 4 字节，6 不是倍数。
module {
  tt.func @mram_offset_not_a_multiple_of_the_element(%p: tensor<64x!tt.ptr<f32>>) {
    %buf = pim.wram_alloc : !pim.memdesc<64xf32, #pim.wram>
    // expected-error @below {{mram_offset 6 is not a multiple of the 4-byte element size}}
    pim.dma_load %p -> %buf {mram_offset = 6 : i64} : tensor<64x!tt.ptr<f32>> -> !pim.memdesc<64xf32, #pim.wram>
    tt.return
  }
}
