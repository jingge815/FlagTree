// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=90 -split-input-file -verify-diagnostics

module attributes {tle.requires_cooperative_grid = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>) attributes {"ttg.global_scratch_memory_alignment" = 4 : i32, "ttg.global_scratch_memory_size" = 4 : i32} {
    tt.return
  }

  tt.func @reject_global_scratch_callee(%mid: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    // expected-error @+2 {{'tle.task_graph.scheduler' op persistent scheduler MVP does not support task callee @producer_body requiring global scratch memory}}
    // expected-error @+1 {{failed to legalize operation 'tle.task_graph.scheduler'}}
    tle.task_graph.scheduler {counter_type = "i32", dispatch = [{callee = @producer_body, task = "producer", task_id = 0 : i64}], edge_consumer_ids = array<i32>, edges = [], initial_ready = ["producer[0]"], initial_ready_ids = array<i32: 0>, instance_coord_offsets = array<i32: 0, 1>, instance_coords = array<i32: 0>, instance_dep_counts = array<i32: 0>, instance_task_ids = array<i32: 0>, instances = [{coord = array<i64: 0>, dep_count = 0 : i64, deps = [], instance = "producer[0]", task = "producer", writes = ["mid[0]"]}], num_edges = 0 : i64, num_instances = 1 : i64, num_tasks = 1 : i64, producer_edge_offsets = array<i32: 0, 0>, queue_capacity = 1 : i64, task_domain_ranks = array<i32: 1>, task_names = ["producer"]}
    tle.task_graph.runtime_state {alignment_bytes = 4 : i64, completed_count_offset_bytes = 16 : i64, counter_bytes = 4 : i64, counter_type = "i32", dep_counters_offset_bytes = 20 : i64, init_flag_offset_bytes = 0 : i64, num_instances = 1 : i64, queue_capacity = 1 : i64, queue_element_bytes = 4 : i64, queue_head_offset_bytes = 8 : i64, queue_lock_offset_bytes = 4 : i64, queue_storage_offset_bytes = 24 : i64, queue_tail_offset_bytes = 12 : i64, state_size_bytes = 28 : i64}
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 1>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
