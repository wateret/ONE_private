nnas_find_package(CaffeProto QUIET)

if (NOT CaffeProto_FOUND)
    return()
endif ()

set(MIR_CAFFE_IMPORTER_SOURCES
        caffe_importer.cpp
        caffe_importer.h
        caffe_op_creator.cpp
        caffe_op_creator.h
        caffe_op_types.h)

add_library(mir_caffe_importer STATIC ${MIR_CAFFE_IMPORTER_SOURCES})
set_target_properties(mir_caffe_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mir_caffe_importer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mir_caffe_importer PUBLIC mir caffeproto PRIVATE stdex)
