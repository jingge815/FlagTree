add_triton_library(TritonPIMTransforms
  ExplicitDMA.cpp
  LowerPIMToEmitC.cpp
  TileToBudget.cpp

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
