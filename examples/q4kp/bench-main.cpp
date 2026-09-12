#include "q4kp_runtime.h"
#include <cstdio>

int llama_bench(int argc, char ** argv);

int main(int argc, char ** argv) {
    const int result = llama_bench(argc, argv);
    std::fprintf(stderr,
        "Q4KP_STATS enabled=%d tensors=%llu metadata_bytes=%llu matrix_ops=%llu selected_tensors=%llu mode=%d\n",
        q4kp_runtime_enabled(),
        (unsigned long long) q4kp_runtime_stat(0),
        (unsigned long long) q4kp_runtime_stat(1),
        (unsigned long long) q4kp_runtime_stat(2),
        (unsigned long long) q4kp_runtime_stat(4), q4kp_runtime_mode());
    return result;
}
