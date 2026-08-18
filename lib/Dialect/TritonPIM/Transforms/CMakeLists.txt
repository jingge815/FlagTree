add_triton_library(TritonPIMTransforms
  ExplicitDMA.cpp

  DEPENDS
  TritonPIMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TritonIR
  TritonPIMIR
)
