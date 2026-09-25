add_triton_library(TritonPIMTransforms
  ExpandPhases.cpp
  ExplicitDMA.cpp
  FuseActivation.cpp
  LowerPIMToEmitC.cpp
  TileToBudget.cpp
  VerifyGmlContract.cpp

  DEPENDS
  TritonPIMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIREmitCDialect
  MLIRSCFDialect
  TritonIR
  TritonPIMIR
)
