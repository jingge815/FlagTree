// RUN: triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-tasklets=16 wram-bytes=65536 mram-bytes=1048576 dma-align=1024' | FileCheck %s

// The hardware description lands on the module, mirroring ttg.num-warps and
// friends. Downstream passes and the GeneSim cost model read it from here.
// CHECK: module attributes {
// CHECK-DAG: "pim.num-dpus" = 1 : i32
// CHECK-DAG: "pim.num-tasklets" = 16 : i32
// CHECK-DAG: pim.target = "pim:v1"
// CHECK-DAG: "pim.wram-bytes" = 65536 : i32
// CHECK-DAG: "pim.mram-bytes" = 1048576 : i64
// CHECK-DAG: "pim.dma-align" = 1024 : i32
// CHECK-LABEL: @elementwise
module {
  tt.func public @elementwise(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>,
                              %c_ptr: !tt.ptr<f32>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %a_ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a_addr = tt.addptr %a_ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b_addr = tt.addptr %b_ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    // Loads stay as tt.load here; pim-explicit-dma lowers them. What changes is
    // that every tensor now carries a tasklet layout.
    // CHECK: tt.load %{{.*}} : tensor<64x!tt.ptr<f32>, #{{.*}}>
    %a = tt.load %a_addr : tensor<64x!tt.ptr<f32>>
    %b = tt.load %b_addr : tensor<64x!tt.ptr<f32>>
    // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<64xf32, #{{.*}}>
    %sum = arith.addf %a, %b : tensor<64xf32>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %c_addr = tt.addptr %c_ptrs, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>, #{{.*}}>
    tt.store %c_addr, %sum : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A 2-D tile with a k-loop: tt.dot survives untouched, since PIM has no tensor
// core imposing an operand layout, and scf.for's iter_args pick up the layout.
// CHECK-LABEL: @matmul_tile
module {
  tt.func public @matmul_tile(%c_ptr: !tt.ptr<f32>, %k: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc_init = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %a = arith.constant dense<1.000000e+00> : tensor<64x32xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<32x128xf16>
    // CHECK: scf.for
    // CHECK-SAME: -> (tensor<64x128xf32, #{{.*}}>)
    %acc = scf.for %i = %c0 to %k step %c1 iter_args(%s = %acc_init)
        -> (tensor<64x128xf32>) : i32 {
      // CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xf16, #{{.*}}> * tensor<32x128xf16, #{{.*}}> -> tensor<64x128xf32, #{{.*}}>
      %d = tt.dot %a, %b, %s : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32>
      scf.yield %d : tensor<64x128xf32>
    }
    %cp = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>>
    tt.store %cp, %acc : tensor<64x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Shape-changing ops route their result encoding through the dialect's layout
// inference interface.
// CHECK-LABEL: @shape_ops
module {
  tt.func public @shape_ops(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    // Built from a load rather than a constant, so that trans is not folded away
    // before it can exercise the layout inference.
    %o = arith.constant dense<0> : tensor<16x32xi32>
    %p = tt.splat %in : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %a = tt.addptr %p, %o : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
    %t = tt.load %a : tensor<16x32x!tt.ptr<f32>>
    // CHECK: tt.trans %{{.*}} : tensor<16x32xf32, #{{.*}}> -> tensor<32x16xf32, #{{.*}}>
    %tr = tt.trans %t {order = array<i32: 1, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    // CHECK: tt.reduce
    %r = "tt.reduce"(%tr) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<32x16xf32>) -> tensor<32xf32>
    %op = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    tt.store %op, %r : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// expand_dims / broadcast, the pattern a 2-D pointer tile is built from.
// CHECK-LABEL: @expand_broadcast
module {
  tt.func public @expand_broadcast(%out: !tt.ptr<i32>) {
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // CHECK: tt.expand_dims %{{.*}} : tensor<64xi32, #{{.*}}> -> tensor<64x1xi32, #{{.*}}>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    // CHECK: tt.broadcast %{{.*}} : tensor<64x1xi32, #{{.*}}> -> tensor<64x32xi32, #{{.*}}>
    %b = tt.broadcast %e : tensor<64x1xi32> -> tensor<64x32xi32>
    %op = tt.splat %out : !tt.ptr<i32> -> tensor<64x32x!tt.ptr<i32>>
    tt.store %op, %b : tensor<64x32x!tt.ptr<i32>>
    tt.return
  }
}
