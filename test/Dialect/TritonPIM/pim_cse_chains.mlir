// RUN: triton-opt %s -split-input-file -pim-expand-phases -pim-cse -pim-verify-gml-contract | FileCheck %s

// 相位链上的 CSE：展开后跑合并，再过契约校验。

// 两个 rope 共享输入，展开后前两相相同、末相不同（Q 链与 K 链）。只合并前缀会把
// 两条链连成一条，契约校验报相位重号，所以整条链不同就一相都不合并。
// CHECK-LABEL: @keep_chains_with_shared_prefix
// CHECK-COUNT-6: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @keep_chains_with_shared_prefix(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %o: tensor<1x32x16x128x!tt.ptr<f16>>,
      %p: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %q = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %k = pim.rope %x, %c, %s {numHeads = 32 : i64, tailCardValue = 3 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %o, %q : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %p, %k : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 两条完全相同的相位链整条合并，只剩一条。
// CHECK-LABEL: @merge_identical_chains
// CHECK-COUNT-3: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_identical_chains(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %o: tensor<1x32x16x128x!tt.ptr<f16>>,
      %p: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %q = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %k = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %o, %q : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %p, %k : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 三条链：第一条只与后两条前缀相同，后两条整条相同。结果不随排列变化，
// 后两条照样合并成一条。
// CHECK-LABEL: @merge_identical_chains_after_prefix_chain
// CHECK-COUNT-6: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_identical_chains_after_prefix_chain(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %oa: tensor<1x32x16x128x!tt.ptr<f16>>, %ob: tensor<1x32x16x128x!tt.ptr<f16>>, %oc: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %ra = pim.rope %x, %c, %s {numHeads = 32 : i64, tailCardValue = 3 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rb = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rc = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %oa, %ra : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %ob, %rb : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %oc, %rc : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 同样三条链换个排列，结果相同。
// CHECK-LABEL: @merge_identical_chains_before_prefix_chain
// CHECK-COUNT-6: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_identical_chains_before_prefix_chain(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %oa: tensor<1x32x16x128x!tt.ptr<f16>>, %ob: tensor<1x32x16x128x!tt.ptr<f16>>, %oc: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %ra = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rb = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rc = pim.rope %x, %c, %s {numHeads = 32 : i64, tailCardValue = 3 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %oa, %ra : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %ob, %rb : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %oc, %rc : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 三条链属性逐相同：a、b 整条相同，c 只有末相前的 sin 不同，所以只与 a、b 共享首相。
// c 排在最前时，a、b 照样整条合并，结果不随排列变化。
// CHECK-LABEL: @merge_identical_chains_after_operand_split
// CHECK-COUNT-6: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_identical_chains_after_operand_split(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %s2: tensor<1x1x16x128xf16>, %oc: tensor<1x32x16x128x!tt.ptr<f16>>, %oa: tensor<1x32x16x128x!tt.ptr<f16>>, %ob: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %rc = pim.rope %x, %c, %s2 {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %ra = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rb = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %oc, %rc : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %oa, %ra : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %ob, %rb : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 同样三条链换个排列，结果相同。
// CHECK-LABEL: @merge_identical_chains_before_operand_split
// CHECK-COUNT-6: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_identical_chains_before_operand_split(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %s2: tensor<1x1x16x128xf16>, %oa: tensor<1x32x16x128x!tt.ptr<f16>>, %ob: tensor<1x32x16x128x!tt.ptr<f16>>, %oc: tensor<1x32x16x128x!tt.ptr<f16>>) {
    %ra = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rb = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    %rc = pim.rope %x, %c, %s2 {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %oa, %ra : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %ob, %rb : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.store %oc, %rc : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}
