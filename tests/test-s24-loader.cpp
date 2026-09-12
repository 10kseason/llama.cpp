#include "llama-model-loader.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

static const char * tensor_name = "blk.0.ffn_gate.weight";

static void check(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void write_fixture(const std::string & path, const std::array<uint8_t, 78> & bytes) {
    ggml_context_ptr ctx(ggml_init({4096, nullptr, false}));
    auto * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_S24, 256, 2);
    ggml_set_name(tensor, tensor_name);
    std::memcpy(tensor->data, bytes.data(), bytes.size());
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "llama");
    gguf_set_val_str(meta.get(), "general.name", "Synthetic S24 loader validation fixture");
    gguf_add_tensor(meta.get(), tensor);
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "fixture write failed");
}

static void run_loader(const std::string & path, llama_load_mode mode, bool invalid,
                       const std::array<uint8_t, 78> & original) {
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, path, splits, nullptr, mode,
                              false, false, true, nullptr, nullptr);
    check(!loader.check_tensors, "test requires optional validation disabled");
    loader.init_mappings(false);
    ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
    auto * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_S24, 256, 2);
    ggml_set_name(tensor, tensor_name);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    check(buffer != nullptr, "CPU buffer allocation failed");
    llama_buf_map buffers;
    bool rejected = false;
    try {
        check(loader.load_all_data(ctx.get(), buffers, nullptr, nullptr, nullptr), "loader returned false");
    } catch (const std::runtime_error & error) {
        const std::string text(error.what());
        rejected = text.find("invalid S24 data") != std::string::npos;
        if (!rejected) { throw; }
    }
    check(rejected == invalid, "mandatory S24 model-load validation mismatch");
    if (!invalid) {
        std::array<uint8_t, 78> readback{};
        ggml_backend_tensor_get(tensor, readback.data(), 0, readback.size());
        check(readback == original, "valid loader payload changed");
    }
    // Exercise the independent ranged-read path with diagnostics still disabled.
    const auto * weight = loader.get_weight(tensor_name);
    check(weight != nullptr, "fixture tensor missing");
    std::array<uint8_t, 78> staging{};
    rejected = false;
    try {
        loader.load_data_range(*weight, 0, staging.size(), staging.data());
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("invalid data") != std::string::npos;
        if (!rejected) { throw; }
    }
    check(rejected == invalid, "mandatory S24 ranged-read validation mismatch");
}

int main() {
    // Fixtures live in the caller's test directory, never a global temp/cache.
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory = "s24-loader-fixture-" + std::to_string(suffix);
    if (!std::filesystem::create_directory(directory)) { return 1; }
    std::vector<std::string> created;
    int result = 0;
    try {
        for (unsigned kind = 0; kind < 3; ++kind) {
            std::array<uint8_t, 78> bytes{};
            if (kind == 1) { bytes[39 + 1] = 0x7c; } // positive infinity scale
            if (kind == 2) {
                const uint64_t invalid = UINT64_C(2821109907456); // exactly 6^16
                for (unsigned byte = 0; byte < 6; ++byte) { bytes[39 + 2 + byte] = (uint8_t) (invalid >> (8 * byte)); }
            }
            const auto path = (directory / (std::to_string(kind) + ".gguf")).string();
            created.push_back(path);
            write_fixture(path, bytes);
            run_loader(path, LLAMA_LOAD_MODE_NONE, kind != 0, bytes);
            run_loader(path, LLAMA_LOAD_MODE_MMAP, kind != 0, bytes);
        }
        std::puts("S24 loader PASS: valid/invalid scale/invalid support across file and mmap paths, including ranged reads; check_tensors=false");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "S24 loader FAIL: %s\n", error.what());
        result = 1;
    }
    for (const auto & path : created) { std::filesystem::remove(path); }
    std::filesystem::remove(directory);
    return result;
}
