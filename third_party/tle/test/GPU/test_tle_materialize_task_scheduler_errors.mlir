// RUN: triton-opt %s --triton-tle-materialize-task-scheduler -split-input-file -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-error @+1 {{requires tle.task_graph.analysis before scheduler materialization}}
  tt.func @reject_missing_analysis(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 1>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {domain_shape = array<i64: 1>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-error @+1 {{requires task graph analysis attribute analysis_version}}
  tt.func @reject_unversioned_analysis(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) attributes {tle.task_graph.analysis = {edges = [], initial_ready = ["producer[0]"], instances = [{dep_count = 0 : i64, deps = [], instance = "producer[0]", task = "producer", writes = ["mid[0]"]}], num_edges = 0 : i64, num_instances = 1 : i64}} {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 1>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {domain_shape = array<i64: 1>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_counter_type() {
    // expected-error @+1 {{MVP supports only counter_type = "i32"}}
    tle.task_graph.scheduler {counter_type = "i64", dispatch = [{task = "producer", task_id = 0 : i64}], edge_consumer_ids = array<i32>, edges = [], initial_ready = ["producer[0]"], initial_ready_ids = array<i32: 0>, instance_coord_offsets = array<i32: 0, 1>, instance_coords = array<i32: 0>, instance_dep_counts = array<i32: 0>, instance_task_ids = array<i32: 0>, instances = [{coord = array<i64: 0>, dep_count = 0 : i64, deps = [], instance = "producer[0]", task = "producer", writes = ["mid[0]"]}], num_edges = 0 : i64, num_instances = 1 : i64, num_tasks = 1 : i64, producer_edge_offsets = array<i32: 0, 0>, queue_capacity = 1 : i64, task_domain_ranks = array<i32: 1>, task_names = ["producer"]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_edge_consumer() {
    // expected-error @+1 {{expects edge consumer to reference instances}}
    tle.task_graph.scheduler {counter_type = "i32", dispatch = [{task = "producer", task_id = 0 : i64}], edge_consumer_ids = array<i32: 0>, edges = [{consumer = "missing[0]", producer = "producer[0]", tile = "mid[0]"}], initial_ready = ["producer[0]"], initial_ready_ids = array<i32: 0>, instance_coord_offsets = array<i32: 0, 1>, instance_coords = array<i32: 0>, instance_dep_counts = array<i32: 0>, instance_task_ids = array<i32: 0>, instances = [{coord = array<i64: 0>, dep_count = 0 : i64, deps = [], instance = "producer[0]", task = "producer", writes = ["mid[0]"]}], num_edges = 1 : i64, num_instances = 1 : i64, num_tasks = 1 : i64, producer_edge_offsets = array<i32: 0, 1>, queue_capacity = 1 : i64, task_domain_ranks = array<i32: 1>, task_names = ["producer"]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_edge_consumer_id() {
    // expected-error @+1 {{expects edge_consumer_ids entries to reference instances}}
    tle.task_graph.scheduler {counter_type = "i32", dispatch = [{task = "producer", task_id = 0 : i64}], edge_consumer_ids = array<i32: 3>, edges = [{consumer = "producer[0]", producer = "producer[0]", tile = "mid[0]"}], initial_ready = ["producer[0]"], initial_ready_ids = array<i32: 0>, instance_coord_offsets = array<i32: 0, 1>, instance_coords = array<i32: 0>, instance_dep_counts = array<i32: 0>, instance_task_ids = array<i32: 0>, instances = [{coord = array<i64: 0>, dep_count = 0 : i64, deps = [], instance = "producer[0]", task = "producer", writes = ["mid[0]"]}], num_edges = 1 : i64, num_instances = 1 : i64, num_tasks = 1 : i64, producer_edge_offsets = array<i32: 0, 1>, queue_capacity = 1 : i64, task_domain_ranks = array<i32: 1>, task_names = ["producer"]}
    tt.return
  }
}
