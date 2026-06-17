// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-stores | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_store_with_known_full_tile_mask
  // CHECK-NOT: ttg.local_load
  // CHECK-NOT: arith.select
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @masked_store_with_known_full_tile_mask(%value: tensor<64x256xbf16, #blocked>) {
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %pid_h = tt.get_program_id z : i32
    %group_h = arith.remsi %pid_h, %c2_i32 : i32
    %h_base = arith.muli %group_h, %c64_i32 : i32
    %h_base_s = tt.splat %h_base : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %h_limit_s = tt.splat %c128_i32 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_h = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %h_idx = arith.addi %h_base_s, %offs_h : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_h_1d = arith.cmpi slt, %h_idx, %h_limit_s : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %mask_h_2d = tt.expand_dims %mask_h_1d {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
    %mask_h = tt.broadcast %mask_h_2d : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %d_limit_s = tt.splat %c512_i32 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_d = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_1d = arith.cmpi slt, %offs_d, %d_limit_s : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_2d = tt.expand_dims %mask_d_1d {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi1, #blocked>
    %mask_d = tt.broadcast %mask_d_2d : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %mask = arith.andi %mask_h, %mask_d : tensor<64x256xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x256xbf16, #shared, #smem, mutable>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.store %ptr, %value, %mask : tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_static_subslice
  tt.func @store_static_subslice(%value: tensor<64x128xbf16, #blocked>) {
    %c128 = arith.constant 128 : i32
    %c128t = tt.splat %c128 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col.off = arith.addi %col, %c128t : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col.off {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %rowb, %colb) : (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: %[[SUB:.*]] = ttg.memdesc_subslice %[[BASE:.*]][0, 128]
    // CHECK-NOT: ttg.local_load
    // CHECK: ttg.local_store %{{.*}}, %[[SUB]]
    // CHECK-NOT: tt.store
    tt.store %ptr, %value : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_store_static_subslice
  tt.func @masked_store_static_subslice(%value: tensor<64x128xbf16, #blocked>) {
    %c127 = arith.constant 127 : i32
    %c128 = arith.constant 128 : i32
    %c128t = tt.splat %c128 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c127t = tt.splat %c127 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask1d = arith.cmpi slt, %col, %c127t : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask2d = tt.expand_dims %mask1d {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi1, #blocked>
    %mask = tt.broadcast %mask2d : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %col.off = arith.addi %col, %c128t : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col.off {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %rowb, %colb) : (!ttg.memdesc<64x512xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: %[[SUB:.*]] = ttg.memdesc_subslice %[[BASE:.*]][0, 128]
    // CHECK: %[[OLD:.*]] = ttg.local_load %[[SUB]]
    // CHECK: %[[MERGED:.*]] = arith.select %{{.*}}, %{{.*}}, %[[OLD]]
    // CHECK: ttg.local_store %[[MERGED]], %[[SUB]]
    // CHECK-NOT: tt.store
    tt.store %ptr, %value, %mask : tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @masked_store_with_partial_tile_mask
  // CHECK: ttg.local_load
  // CHECK: arith.select
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.store
  tt.func @masked_store_with_partial_tile_mask(%value: tensor<64x256xbf16, #blocked>) {
    %c255_i32 = arith.constant 255 : i32
    %d_limit_s = tt.splat %c255_i32 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_d = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_1d = arith.cmpi slt, %offs_d, %d_limit_s : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_d_2d = tt.expand_dims %mask_d_1d {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi1, #blocked>
    %mask = tt.broadcast %mask_d_2d : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem) : (!ttg.memdesc<64x256xbf16, #shared, #smem, mutable>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.store %ptr, %value, %mask : tensor<64x256x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}
