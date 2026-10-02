// RUN: not triton-opt %s -pim-verify-gml-contract 2>&1 | FileCheck %s

// The hardware half of the placement check has to run on the operator-level
// chain too.
//
// `verifyModulePlacement` (placement spread vs the device's DPU count, and the
// named DPU ids) lived only at the exit of `-convert-triton-to-pim`. The B path
// does not pass through it, so a placement over 4 DPUs naming ids 7/9/11 was
// accepted on a 2-DPU device with no diagnostic -- measured, zero warnings.
// The check now also runs here.
//
// Note the module states its own `pim.num-dpus`: the fix must read that value,
// not the pass-option default, or "not declared" would be treated as a hardware
// fact.

// CHECK: the device has only 2
module attributes {"pim.num-dpus" = 2 : i32, pim.placement = #pim.placement<kind = shard, dim = 0, numDpus = 4, dpuIds = [0, 7, 9, 11]>} {
  tt.func public @four_dpus_on_a_two_dpu_device() {
    tt.return
  }
}
