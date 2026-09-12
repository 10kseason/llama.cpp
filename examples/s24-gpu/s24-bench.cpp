// SPDX-License-Identifier: MIT
// Bounded S24/F32 MUL_MAT through the real ggml Vulkan backend. No CPU fallback.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-s24.h"
#include "ggml-vulkan.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct options {
    bool self_test = false;
    std::string model, tensor, negative;
    uint32_t device = 0, inputs = 1, warmup = 5, repetitions = 20;
};

static uint32_t number(const std::string & text, uint32_t low, uint32_t high) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos, "Expected unsigned integer");
    size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    require(consumed == text.size() && value >= low && value <= high, "Numeric option outside bounded range");
    return uint32_t(value);
}

static void usage() {
    std::cout << "llama-s24-bench --self-test [--device 0]\n"
                 "llama-s24-bench --model MODEL.gguf --tensor NAME [--inputs 1] [--device 0]\n"
                 "                [--warmup 5] [--repetitions 20]\n"
                 "Inputs: 1..1024; warmup: 0..1000; repetitions: 1..1000.\n"
                 "Times are synchronized graph wall time, not GPU timestamp intervals or model tokens/s.\n";
}

static options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--self-test") {
            result.self_test = true;
            continue;
        }
        require(i + 1 < argc, "Missing option value");
        const std::string value = argv[++i];
        if      (arg == "--model")       { result.model = value; }
        else if (arg == "--tensor")      { result.tensor = value; }
        else if (arg == "--negative")    { result.negative = value; }
        else if (arg == "--device")      { result.device = number(value, 0, GGML_VK_MAX_DEVICES - 1); }
        else if (arg == "--inputs")      { result.inputs = number(value, 1, 1024); }
        else if (arg == "--warmup")      { result.warmup = number(value, 0, 1000); }
        else if (arg == "--repetitions") { result.repetitions = number(value, 1, 1000); }
        else { throw std::runtime_error("Unknown option: " + arg); }
    }
    const bool file_mode = !result.model.empty() || !result.tensor.empty();
    require(int(result.self_test) + int(file_mode) + int(!result.negative.empty()) == 1, "Choose self-test, model/tensor, or negative-test mode");
    require(!file_mode || (!result.model.empty() && !result.tensor.empty()), "Model and tensor are required together");
    return result;
}

struct matrix {
    int64_t columns = 0, rows = 0;
    std::vector<uint8_t> compact;
};

static size_t checked_shape(int64_t columns, int64_t rows, uint32_t inputs) {
    // Conservative host bounds complement the backend's actual device limits.
    // They also keep every shader address/product in its uint32 indexing range.
    require(columns > 0 && columns <= 262144 && columns % 256 == 0, "Columns must be a positive multiple of 256, at most 262144");
    require(rows >= 2 && rows <= 65535 && inputs >= 1 && inputs <= 1024, "Rows/input count outside supported 2D limits");
    const uint64_t bytes = uint64_t(rows) * uint64_t(columns / 256) * 39;
    require(bytes <= 256ull * 1024 * 1024, "Selected S24 tensor exceeds 256 MiB tool limit");
    require(uint64_t(columns) * inputs * sizeof(float) <= 32ull * 1024 * 1024, "Input buffer exceeds 32 MiB tool limit");
    require(uint64_t(rows) * inputs * sizeof(float) <= 32ull * 1024 * 1024, "Output buffer exceeds 32 MiB tool limit");
    require(uint64_t(rows) * uint64_t(columns) * inputs <= 512ull * 1024 * 1024, "FP64 verification workload exceeds 512 million terms");
    return size_t(bytes);
}

static matrix load_matrix(const options & opt) {
    gguf_init_params params{true, nullptr};
    gguf_ptr info(gguf_init_from_file(opt.model.c_str(), params), gguf_free);
    require(bool(info), "Cannot read GGUF metadata");
    const int64_t index = gguf_find_tensor(info.get(), opt.tensor.c_str());
    require(index >= 0, "Tensor not found in GGUF");
    require(gguf_get_tensor_type(info.get(), index) == GGML_TYPE_S24, "Selected tensor is not this fork's S24 type 63");
    const auto * ne = gguf_get_tensor_ne(info.get(), index);
    require(ne[2] == 1 && ne[3] == 1, "Only a complete contiguous 2D weight tensor is supported");
    matrix result;
    result.columns = ne[0];
    result.rows = ne[1];
    const size_t bytes = checked_shape(result.columns, result.rows, opt.inputs);
    require(gguf_get_tensor_size(info.get(), index) == bytes, "GGUF S24 size mismatch");
    const uint64_t data_offset = gguf_get_data_offset(info.get());
    const uint64_t tensor_offset = gguf_get_tensor_offset(info.get(), index);
    require(tensor_offset <= UINT64_MAX - data_offset, "Tensor offset overflow");
    const uint64_t offset = data_offset + tensor_offset;
    std::ifstream input(std::filesystem::u8path(opt.model), std::ios::binary | std::ios::ate);
    require(bool(input), "Cannot open GGUF data");
    const auto end = input.tellg();
    require(end >= 0 && offset <= uint64_t(end) && bytes <= uint64_t(end) - offset, "Tensor data exceeds GGUF file bounds");
    require(offset <= uint64_t(std::numeric_limits<std::streamoff>::max()), "Tensor offset exceeds stream range");
    result.compact.resize(bytes);
    input.seekg(std::streamoff(offset));
    input.read(reinterpret_cast<char *>(result.compact.data()), std::streamsize(bytes));
    require(bool(input), "Truncated GGUF tensor read");
    require(ggml_s24_validate_compact(result.compact.data(), bytes), "Invalid S24 compact payload");
    return result;
}

static uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static matrix synthetic(int64_t columns, int64_t rows) {
    matrix result;
    result.columns = columns;
    result.rows = rows;
    result.compact.resize(checked_shape(columns, rows, 1));
    const size_t blocks = result.compact.size() / 39;
    std::vector<uint8_t> direct(blocks * 42, 0);
    uint32_t state = 240125;
    // Zero, minimum subnormal/normal half, and ordinary finite scales.
    const uint16_t scales[] = {0, 1, 0x0400, 0x2c00, 0x3000, 0x3c00};
    for (size_t b = 0; b < blocks; ++b) {
        const auto scale = scales[b % 6];
        auto * block = direct.data() + b * 42;
        block[0] = uint8_t(scale);
        block[1] = uint8_t(scale >> 8);
        for (unsigned group = 0; group < 64; ++group) {
            const unsigned bit = group * 3;
            const unsigned mask = unsigned((b + group) % 6);
            block[2 + bit / 8] |= uint8_t(mask << (bit % 8));
            if (bit % 8 > 5) {
                block[3 + bit / 8] |= uint8_t(mask >> (8 - bit % 8));
            }
            block[26 + group / 4] |= uint8_t((next_random(state) & 3) << (2 * (group % 4)));
        }
    }
    require(ggml_s24_direct42_to_compact(direct.data(), direct.size(), result.compact.data(), result.compact.size()), "Synthetic encoding failed");
    return result;
}

static std::vector<float> make_inputs(int64_t columns, uint32_t count) {
    std::vector<float> result(size_t(columns) * count);
    uint32_t state = 240125;
    for (auto & value : result) {
        // Exact normal F32 fractions; no libstdc++ random distribution drift.
        value = float(int(next_random(state) % 1025) - 512) / 512.0f;
    }
    return result;
}

struct graph_data {
    context_ptr weights_context{nullptr, ggml_free}, work_context{nullptr, ggml_free};
    buffer_ptr weights_buffer{nullptr, ggml_backend_buffer_free}, work_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * weights = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
};

static graph_data allocate_graph(ggml_backend_t backend, const matrix & mat, uint32_t count) {
    checked_shape(mat.columns, mat.rows, count);
    graph_data data;
    ggml_init_params weight_params{ggml_tensor_overhead() * 8, nullptr, true};
    ggml_init_params work_params{ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true};
    data.weights_context.reset(ggml_init(weight_params));
    data.work_context.reset(ggml_init(work_params));
    require(data.weights_context && data.work_context, "Context allocation failed");
    data.weights = ggml_new_tensor_2d(data.weights_context.get(), GGML_TYPE_S24, mat.columns, mat.rows);
    ggml_set_name(data.weights, "s24_weights");
    data.input = ggml_new_tensor_2d(data.work_context.get(), GGML_TYPE_F32, mat.columns, count);
    ggml_set_input(data.input);
    data.output = ggml_mul_mat(data.work_context.get(), data.weights, data.input);
    ggml_set_output(data.output);
    require(ggml_backend_supports_op(backend, data.output), "Vulkan backend does not support this S24/F32 shape");
    data.graph = ggml_new_graph_custom(data.work_context.get(), 16, false);
    ggml_build_forward_expand(data.graph, data.output);
    data.weights_buffer.reset(ggml_backend_alloc_ctx_tensors(data.weights_context.get(), backend));
    data.work_buffer.reset(ggml_backend_alloc_ctx_tensors(data.work_context.get(), backend));
    require(data.weights_buffer && data.work_buffer, "Backend buffer allocation failed");
    ggml_backend_buffer_set_usage(data.weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return data;
}

static void compute(ggml_backend_t backend, graph_data & data) {
    require(ggml_backend_graph_compute(backend, data.graph) == GGML_STATUS_SUCCESS, "Vulkan graph execution failed");
    ggml_backend_synchronize(backend);
}

static std::vector<float> read_output(graph_data & data) {
    const size_t count = size_t(ggml_nelements(data.output));
    // These guards check the host readback's bounds, not device buffer bounds.
    std::vector<float> guarded(count + 16, 123456.0f);
    ggml_backend_tensor_get(data.output, guarded.data() + 8, 0, count * sizeof(float));
    for (size_t i = 0; i < 8; ++i) {
        require(guarded[i] == 123456.0f && guarded[count + 8 + i] == 123456.0f, "Host output readback guard changed");
    }
    return {guarded.begin() + 8, guarded.end() - 8};
}

static json verify_reference(const matrix & mat, const std::vector<float> & inputs,
                             uint32_t count, const std::vector<float> & output) {
    std::vector<float> row(size_t(mat.columns), 0.0f);
    double maximum_error = 0, maximum_ratio = 0;
    const double unit_roundoff = std::ldexp(1.0, -24);
    // At most 32 products per lane/chunk, separate product/add rounding, then
    // five shared reduction additions. This conservative forward-error bound
    // scales with each output's sum of absolute FP64 terms, not output size.
    const double operations = 64.0 * std::ceil(double(mat.columns / 256) / 8.0) + 5.0;
    const double gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff);
    for (int64_t r = 0; r < mat.rows; ++r) {
        dequantize_row_s24(mat.compact.data() + size_t(r) * size_t(mat.columns / 256) * 39, row.data(), mat.columns);
        for (uint32_t v = 0; v < count; ++v) {
            double expected = 0, absolute_sum = 0;
            for (int64_t k = 0; k < mat.columns; ++k) {
                const double term = double(row[size_t(k)]) * double(inputs[size_t(v) * size_t(mat.columns) + size_t(k)]);
                expected += term;
                absolute_sum += std::abs(term);
            }
            const double actual = output[size_t(v) * size_t(mat.rows) + size_t(r)];
            const double error = std::abs(actual - expected);
            const double bound = gamma * absolute_sum;
            require(std::isfinite(actual) && error <= bound, "GPU output exceeds FP64 forward-error bound");
            maximum_error = std::max(maximum_error, error);
            maximum_ratio = std::max(maximum_ratio, bound == 0 ? 0 : error / bound);
        }
    }
    return {{"outputs_checked", output.size()}, {"max_absolute_error", maximum_error},
            {"max_error_bound_ratio", maximum_ratio}, {"bound_gamma", gamma},
            {"reference", "CPU compact dequantization, FP64 products and sum"}};
}

static void check_roundtrip(graph_data & data, const matrix & mat) {
    std::vector<uint8_t> got(mat.compact.size(), 0);
    ggml_backend_tensor_get(data.weights, got.data(), 0, got.size());
    require(got == mat.compact, "Compact upload/readback roundtrip changed bytes");
    // The backend may serve partial reads by reversing the entire expansion.
    const size_t offset = 7, length = std::min<size_t>(31, got.size() - offset);
    std::vector<uint8_t> partial(length, 0);
    ggml_backend_tensor_get(data.weights, partial.data(), offset, length);
    require(std::equal(partial.begin(), partial.end(), mat.compact.begin() + offset), "Partial compact readback mismatch");
}

static json run_matrix(ggml_backend_t backend, const matrix & mat, const options & opt) {
    auto data = allocate_graph(backend, mat, opt.inputs);
    const auto inputs = make_inputs(mat.columns, opt.inputs);
    ggml_backend_tensor_set(data.weights, mat.compact.data(), 0, mat.compact.size());
    ggml_backend_tensor_set(data.input, inputs.data(), 0, inputs.size() * sizeof(float));
    check_roundtrip(data, mat);
    compute(backend, data);
    const auto before = read_output(data);
    auto correctness = verify_reference(mat, inputs, opt.inputs, before);
    for (uint32_t i = 0; i < opt.warmup; ++i) {
        compute(backend, data);
    }
    std::vector<double> samples;
    samples.reserve(opt.repetitions);
    for (uint32_t i = 0; i < opt.repetitions; ++i) {
        ggml_backend_synchronize(backend);
        const auto start = clock_type::now();
        compute(backend, data);
        samples.push_back(std::chrono::duration<double, std::micro>(clock_type::now() - start).count());
    }
    const auto after = read_output(data);
    require(before.size() == after.size() && std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0, "Repeated graph output changed bits");
    check_roundtrip(data, mat);
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5;
    double variance = 0;
    for (const auto sample : samples) { variance += (sample - mean) * (sample - mean); }
    const double sd = samples.size() > 1 ? std::sqrt(variance / (samples.size() - 1)) : 0;
    correctness["compact_roundtrip"] = true;
    correctness["partial_compact_readback"] = true;
    correctness["repeated_output_bit_identical"] = true;
    correctness["host_readback_guards"] = true;
    return {{"status", "PASS"}, {"columns", mat.columns}, {"rows", mat.rows}, {"inputs", opt.inputs},
            {"input_seed", 240125}, {"input_type", "F32"}, {"weight_type", "S24"}, {"weight_type_id", int(GGML_TYPE_S24)},
            {"compact_bytes", mat.compact.size()}, {"direct42_payload_bytes", mat.compact.size() / 39 * 42},
            {"weight_buffer_allocation_bytes", ggml_backend_buffer_get_size(data.weights_buffer.get())},
            {"working_buffer_allocation_bytes", ggml_backend_buffer_get_size(data.work_buffer.get())},
            {"correctness", correctness},
            {"timing", {{"scope", "synchronized ggml Vulkan graph wall time; not device timestamps or model tokens/s"},
                        {"unit", "microseconds"}, {"validation_dispatches_before_warmup", 1}, {"warmup", opt.warmup},
                        {"repetitions", opt.repetitions}, {"mean", mean}, {"median", median}, {"sample_sd", sd},
                        {"samples", samples}, {"excludes", {"upload", "initial graph execution", "readback", "FP64 reference"}}}}};
}

static int negative_test(ggml_backend_t backend, const options & opt) {
    require(opt.negative == "invalid-scale" || opt.negative == "invalid-support" ||
            opt.negative == "partial-upload" || opt.negative == "view", "Unknown negative-test case");
    const auto mat = synthetic(256, 4);
    auto data = allocate_graph(backend, mat, 1);
    ggml_backend_tensor_set(data.weights, mat.compact.data(), 0, mat.compact.size());
    check_roundtrip(data, mat);
    // Subprocess tests require this marker AND the backend assertion. A device
    // initialization failure must never count as successful input rejection.
    std::cout << json({{"type", "rejection_probe_ready"}, {"case", opt.negative}}).dump() << std::endl;
    auto invalid = mat.compact;
    if (opt.negative == "invalid-scale") {
        invalid[0] = 0; invalid[1] = 0x7c;
        ggml_backend_tensor_set(data.weights, invalid.data(), 0, invalid.size());
    } else if (opt.negative == "invalid-support") {
        std::fill(invalid.begin() + 2, invalid.begin() + 23, 0xff);
        ggml_backend_tensor_set(data.weights, invalid.data(), 0, invalid.size());
    } else if (opt.negative == "partial-upload") {
        ggml_backend_tensor_set(data.weights, invalid.data(), 0, invalid.size() - 1);
    } else {
        auto * view = ggml_view_2d(data.weights_context.get(), data.weights, 256, 2,
                                   ggml_row_size(GGML_TYPE_S24, 256), 0);
        (void) ggml_backend_view_init(view);
    }
    std::cout << json({{"status", "FAIL"}, {"error", "Backend accepted forbidden operation"}}).dump() << std::endl;
    return 2;
}

static json check_unsupported_ops(ggml_backend_t backend) {
    ggml_init_params params{ggml_tensor_overhead() * 16, nullptr, true};
    context_ptr ctx(ggml_init(params), ggml_free);
    require(bool(ctx), "Support-query context allocation failed");
    auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_S24, 256, 4);
    auto * duplicate = ggml_dup_tensor(ctx.get(), weight);
    auto * concat = ggml_concat(ctx.get(), weight, weight, 1);
    auto * view = ggml_view_2d(ctx.get(), weight, 256, 2, ggml_row_size(GGML_TYPE_S24, 256), 0);
    auto * copy = ggml_cpy(ctx.get(), weight, duplicate);
    auto * storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
    // A one-byte offset is invalid even on a device permitting small descriptor
    // offsets: the shader's float load still requires float element alignment.
    auto * misaligned = ggml_view_2d(ctx.get(), storage, 256, 1, 256 * sizeof(float), 1);
    auto * matmul = ggml_mul_mat(ctx.get(), weight, misaligned);
    const std::vector<std::pair<const char *, ggml_tensor *>> cases = {
        {"S24_CONCAT", concat}, {"S24_VIEW", view}, {"S24_CPY", copy}, {"misaligned_F32_input", matmul}
    };
    json passed = json::array();
    for (const auto & item : cases) {
        require(!ggml_backend_supports_op(backend, item.second), "Backend advertised an unsupported S24 operation");
        passed.push_back(item.first);
    }
    return passed;
}

static int run_main(int argc, char ** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            usage();
            return argc == 1 ? 1 : 0;
        }
        const auto opt = parse(argc, argv);
        require(int(opt.device) < ggml_backend_vk_get_device_count(), "Requested Vulkan device is unavailable");
        backend_ptr backend(ggml_backend_vk_init(opt.device), ggml_backend_free);
        require(bool(backend) && ggml_backend_is_vk(backend.get()), "Vulkan backend initialization failed");
        if (!opt.negative.empty()) { return negative_test(backend.get(), opt); }
        json report = {{"tool", "llama-s24-bench"}, {"backend", ggml_backend_name(backend.get())},
                       {"device_index", opt.device}, {"cpu_fallback", false}};
        if (opt.self_test) {
            report["mode"] = "self-test";
            report["unsupported_ops_rejected"] = check_unsupported_ops(backend.get());
            report["cases"] = json::array();
            for (const auto & shape : std::vector<std::vector<int64_t>>{{256, 7, 1}, {512, 17, 3}, {2304, 9, 2}}) {
                options test = opt;
                test.inputs = uint32_t(shape[2]);
                test.warmup = 1;
                test.repetitions = 2;
                report["cases"].push_back(run_matrix(backend.get(), synthetic(shape[0], shape[1]), test));
            }
        } else {
            report["mode"] = "gguf-tensor";
            report["tensor"] = opt.tensor;
            report["result"] = run_matrix(backend.get(), load_matrix(opt), opt);
        }
        report["status"] = "PASS";
        std::cout << report.dump(2) << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cout << json({{"status", "FAIL"}, {"error", error.what()}}).dump() << std::endl;
        return 1;
    }
}

#ifdef _WIN32
// Preserve arbitrary Windows model paths; the gguf reader takes UTF-8 and the
// selected payload stream above converts UTF-8 back through filesystem::u8path.
int wmain(int argc, wchar_t ** argv);
int wmain(int argc, wchar_t ** argv) {
    try {
        std::vector<std::string> storage;
        storage.reserve(size_t(argc));
        for (int i = 0; i < argc; ++i) {
            const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
            require(bytes > 0, "Invalid Windows Unicode argument");
            std::string value(size_t(bytes), '\0');
            require(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), bytes, nullptr, nullptr) == bytes,
                    "Windows Unicode conversion failed");
            value.resize(size_t(bytes) - 1);
            storage.push_back(std::move(value));
        }
        std::vector<char *> utf8;
        utf8.reserve(size_t(argc));
        for (auto & value : storage) { utf8.push_back(value.data()); }
        return run_main(argc, utf8.data());
    } catch (const std::exception & error) {
        std::cout << json({{"status", "FAIL"}, {"error", error.what()}}).dump() << std::endl;
        return 1;
    }
}
#else
int main(int argc, char ** argv) { return run_main(argc, argv); }
#endif
