foreach(tool main llama-bench perplexity ifairy-actq-microbench ifairy-vecdot-microbench)
    add_subdirectory(${tool})
endforeach()

if (GGML_LEGACY_IFAIRY_CPU_LUT)
    add_subdirectory(ifairy-microbench)
endif()
