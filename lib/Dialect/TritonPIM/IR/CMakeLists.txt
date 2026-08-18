add_triton_library(TritonPIMIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonPIMTableGen
  TritonPIMAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
