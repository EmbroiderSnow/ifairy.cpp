llama_add_compile_flags()

function(ifairy_test name)
    add_executable(${name} ${name}.cpp get-model.cpp)
    target_link_libraries(${name} PRIVATE common)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

ifairy_test(test-legacy-ifairy-direct)
ifairy_test(test-ifairy-model)
add_executable(test-backend-ops test-backend-ops.cpp get-model.cpp)
target_link_libraries(test-backend-ops PRIVATE common)
set_property(TEST test-ifairy-model PROPERTY ENVIRONMENT "GGML_IFAIRY_LUT=0")
add_test(NAME test-ifairy-ops COMMAND test-backend-ops test -b CPU
    -o IFAIRY_ADD,IFAIRY_MUL,IFAIRY_RMS_NORM,IFAIRY_SPLIT,IFAIRY_MERGE,IFAIRY_RELU2,IFAIRY_ROPE)

if (GGML_LEGACY_IFAIRY_CPU_LUT)
    ifairy_test(test-legacy-ifairy)
    target_compile_definitions(test-legacy-ifairy PRIVATE GGML_USE_LEGACY_IFAIRY_CPU_LUT)
    set_property(TEST test-legacy-ifairy PROPERTY ENVIRONMENT "GGML_IFAIRY_LUT=1")
    add_test(NAME test-ifairy-model-lut COMMAND test-ifairy-model)
    set_property(TEST test-ifairy-model-lut PROPERTY ENVIRONMENT "GGML_IFAIRY_LUT=1")
endif()
