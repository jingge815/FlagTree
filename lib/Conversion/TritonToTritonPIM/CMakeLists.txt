add_triton_library(TritonToTritonPIM
    TritonPIMConversion.cpp
    TritonToTritonPIMPass.cpp

    DEPENDS
    TritonPIMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonPIMIR
)
