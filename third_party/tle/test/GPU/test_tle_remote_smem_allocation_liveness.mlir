// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remote_exposed_smem_lives_until_next_distributed_barrier
  // CHECK: %[[ALLOC:.+]] = ttg.local_alloc {allocation.offset = 0 : i32
  // CHECK: "tle.remote_pointers"
  // CHECK: tle.exclusive_cumsum
  // CHECK-SAME: allocation.offset = 4096 : i32
  // CHECK: tle.distributed_barrier
  tt.func @remote_exposed_smem_lives_until_next_distributed_barrier(%out: !tt.ptr<i32>) {
    %c1_i32 = arith.constant 1 : i32
    %offs = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %values = arith.constant dense<1> : tensor<512xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #smem, mutable>
    %ptrs = "tle.local_pointers"(%alloc, %offs) : (!ttg.memdesc<1024xi32, #shared, #smem, mutable>, tensor<512xi32, #blocked>) -> tensor<512x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptrs, %values : tensor<512x!tt.ptr<i32, 3>, #blocked>
    tle.distributed_barrier
    %remote_ptrs = "tle.remote_pointers"(%ptrs, %c1_i32) : (tensor<512x!tt.ptr<i32, 3>, #blocked>, i32) -> tensor<512x!tt.ptr<i32, 7>, #blocked>
    %remote = tt.load %remote_ptrs : tensor<512x!tt.ptr<i32, 7>, #blocked>
    %exclusive, %total = "tle.exclusive_cumsum"(%remote) {axis = 0 : i32, reverse = false} : (tensor<512xi32, #blocked>) -> (tensor<512xi32, #blocked>, i32)
    %out_ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>, #blocked>
    %out_ptrs_off = tt.addptr %out_ptrs, %offs : tensor<512x!tt.ptr<i32>, #blocked>, tensor<512xi32, #blocked>
    tt.store %out_ptrs_off, %exclusive : tensor<512x!tt.ptr<i32>, #blocked>
    tle.distributed_barrier
    tt.return
  }
}
