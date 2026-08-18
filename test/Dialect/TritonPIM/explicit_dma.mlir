// RUN: triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-tasklets=16 wram-bytes=65536' -pim-explicit-dma | FileCheck %s

// A load becomes: allocate a WRAM buffer, DMA into it, barrier, then read it.
// The barrier is not optional -- the transfer is asynchronous with respect to
// the tasklets that consume the data.
// CHECK-LABEL: @single_load
// CHECK: %[[BUF:.*]] = pim.wram_alloc : !pim.memdesc<64xf32, #pim.wram>
// CHECK: pim.dma_load %{{.*}} -> %[[BUF]]
// CHECK: pim.barrier
// CHECK: pim.wram_load %[[BUF]]
// tt.load must be gone entirely: an implicit global access has no meaning on a
// device with no cache.
// CHECK-NOT: tt.load
module {
  tt.func public @single_load(%a_ptr: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %a = tt.load %addr : tensor<64x!tt.ptr<f32>>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %op, %a : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A store becomes: stage into WRAM, barrier so every tasklet has contributed,
// then DMA out.
// CHECK-LABEL: @single_store
// CHECK: %[[BUF:.*]] = pim.wram_alloc : !pim.memdesc<64xf32, #pim.wram>
// CHECK: pim.wram_store %{{.*}}, %[[BUF]]
// CHECK: pim.barrier
// CHECK: pim.dma_store %[[BUF]] -> %{{.*}}
// CHECK-NOT: tt.store
module {
  tt.func public @single_store(%c_ptr: !tt.ptr<f32>) {
    %v = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %ptrs = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %addr, %v : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The pointer analysis proves a unit stride for a plain make_range offset, and
// traces the base back to argument 0.
// CHECK-LABEL: @proven_contiguous
// CHECK: pim.dma_load
// CHECK-SAME: base_arg = 0 : i64
// CHECK-SAME: contiguous_dim = 0 : i64
// CHECK-SAME: elem_stride = 1 : i64
module {
  tt.func public @proven_contiguous(%a_ptr: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %a = tt.load %addr : tensor<64x!tt.ptr<f32>>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %op, %a : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A 2-D row-major tile: the last dimension is unit-stride, which is the one a
// DMA engine can walk contiguously.
// CHECK-LABEL: @proven_2d
// CHECK: pim.dma_load
// CHECK-SAME: contiguous_dim = 1 : i64
// CHECK-SAME: elem_stride = 1 : i64
module {
  tt.func public @proven_2d(%a_ptr: !tt.ptr<f16>, %o: !tt.ptr<f16>) {
    %cst = arith.constant dense<32> : tensor<64x1xi32>
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %rk = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %em = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %rows = arith.muli %em, %cst : tensor<64x1xi32>
    %bm = tt.broadcast %rows : tensor<64x1xi32> -> tensor<64x32xi32>
    %ek = tt.expand_dims %rk {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %bk = tt.broadcast %ek : tensor<1x32xi32> -> tensor<64x32xi32>
    %off = arith.addi %bm, %bk : tensor<64x32xi32>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    %addr = tt.addptr %ptrs, %off : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
    %a = tt.load %addr : tensor<64x32x!tt.ptr<f16>>
    %op = tt.splat %o : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    tt.store %op, %a : tensor<64x32x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// A gather through indices read from memory cannot be proven strided. The
// attributes must be absent rather than guessed: their absence is what tells a
// backend it may not assume a layout.
//
// The index load itself IS strided, so the first dma_load is annotated; the
// gather that consumes those indices is the one that must not be.
// CHECK-LABEL: @unproven_gather
// CHECK: pim.dma_load %{{.*}} -> %{{.*}} {base_arg
// CHECK: pim.dma_load %{{.*}} -> %{{[0-9]+}} :
// CHECK-NOT: contiguous_dim
module {
  tt.func public @unproven_gather(%a_ptr: !tt.ptr<f32>, %i_ptr: !tt.ptr<i32>,
                                  %o: !tt.ptr<f32>) {
    // Indices read from memory: opaque to the analysis, so nothing is provable.
    %ir = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %ip = tt.splat %i_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %ia = tt.addptr %ip, %ir : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    %idx = tt.load %ia : tensor<64x!tt.ptr<i32>>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %a = tt.load %addr : tensor<64x!tt.ptr<f32>>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %op, %a : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The staging buffer is allocated once, outside the loop, not once per
// iteration: WRAM is far too small to afford per-tile allocation.
// CHECK-LABEL: @alloc_hoisted_out_of_loop
// CHECK: pim.wram_alloc
// CHECK: scf.for
// CHECK-NOT: pim.wram_alloc
// CHECK: pim.dma_load
module {
  tt.func public @alloc_hoisted_out_of_loop(%a_ptr: !tt.ptr<f32>, %n: i32,
                                            %o: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%s = %init)
        -> (tensor<64xf32>) : i32 {
      %a = tt.load %addr : tensor<64x!tt.ptr<f32>>
      %sum = arith.addf %s, %a : tensor<64xf32>
      scf.yield %sum : tensor<64xf32>
    }
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %op, %acc : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// WRAM usage is reported on the module. Two f16 tiles, 64x32 and 32x128:
// 4096 + 8192 = 12288 bytes, well inside the 64 KiB budget.
// CHECK: "pim.wram-bytes-used" = 12288 : i32
// CHECK-LABEL: @wram_usage_reported
module {
  tt.func public @wram_usage_reported(%a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>) {
    %ao = arith.constant dense<0> : tensor<64x32xi32>
    %bo = arith.constant dense<0> : tensor<32x128xi32>
    %ap = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    %aa = tt.addptr %ap, %ao : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
    %bp = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
    %ba = tt.addptr %bp, %bo : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
    %a = tt.load %aa : tensor<64x32x!tt.ptr<f16>>
    %b = tt.load %ba : tensor<32x128x!tt.ptr<f16>>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %d = tt.dot %a, %b, %acc : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32>
    tt.return
  }
}

// -----

// A masked load keeps its mask and fill value on the DMA: lanes the mask
// excludes are filled rather than transferred.
// CHECK-LABEL: @masked_load
// CHECK: pim.dma_load %{{[0-9]+}}, %{{[0-9]+}}, %{{[a-z0-9_]+}} -> %{{[0-9]+}}
module {
  tt.func public @masked_load(%a_ptr: !tt.ptr<f32>, %n: i32, %o: !tt.ptr<f32>) {
    %fill = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %lim = tt.splat %n : i32 -> tensor<64xi32>
    %mask = arith.cmpi slt, %range, %lim : tensor<64xi32>
    %ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %a = tt.load %addr, %mask, %fill : tensor<64x!tt.ptr<f32>>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %op, %a : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
