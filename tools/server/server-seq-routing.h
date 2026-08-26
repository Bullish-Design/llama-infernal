#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// P2 fork: per-sequence LoRA routing for llama-server (--lora-seq-routing).
//
// One llama_decode carries a mix of adapters: the server registers a fixed
// adapter pool once at startup and routes each slot to one pool index per
// decode step. The checks below are the startup contract. Every one of them was
// observed by a probe, not derived; the evidence is in
// .scratch/projects/005-server-seq-routing/spikes/S11/interactions.md section 8.

// The three fork entry points, resolved at run time. A server built with this
// header still starts against a libllama that exports none of them, so R-1 can
// name the missing symbol instead of the process dying at dynamic link.
struct seq_routing_api {
    int32_t (*set_seq_adapters)      (llama_context *, llama_adapter_lora **, size_t)  = nullptr;
    int32_t (*set_seq_adapter)       (llama_context *, llama_seq_id, int32_t)          = nullptr;
    int32_t (*set_seq_adapter_scaled)(llama_context *, llama_seq_id, int32_t, float)   = nullptr;

    std::vector<std::string> missing; // names the loaded library does not export

    bool complete() const { return missing.empty(); }

    static seq_routing_api resolve();
};

// One pool adapter, read from its own GGUF at startup.
struct seq_routing_adapter {
    std::string path;
    size_t      n_pairs  = 0;     // LoRA A/B pairs, named by W-2
    size_t      n_bytes  = 0;     // W-4: tensor data, summed from the GGUF
    bool        is_alora = false; // R-7
    std::string moe_tensor;       // R-5: first expert weight this adapter targets, empty when clean
    bool        gguf_ok  = false; // false = the GGUF could not be read
};

struct seq_routing_config {
    // library
    std::vector<std::string> missing_symbols;    // R-1
    bool        node_budget_fixed = false;       // R-2 probe result
    std::string build_info;

    // context and model
    int32_t  n_parallel = 0;                     // R-6, W-1
    int32_t  n_seq_max  = 0;                     // R-6
    uint32_t n_ubatch   = 0;                     // R-2: the graph_max_nodes input
    int32_t  n_tensors  = 0;                     // R-2: 0 = unknown

    // interactions
    bool        hats_registered  = false;        // R-3
    int32_t     cache_ram_mib    = 0;            // R-4
    bool        cache_idle_slots = false;        // R-4
    std::string slot_save_path;                  // R-4
    bool        has_draft_model  = false;        // W-3

    // device memory, W-4. Both 0 = unknown (no offload device, or the backend
    // could not be asked); W-4 then falls back and says so.
    size_t   vram_free_bytes  = 0;
    size_t   vram_total_bytes = 0;

    std::vector<seq_routing_adapter> pool;
};

struct seq_routing_finding {
    std::string id;      // "R-2", "W-1"
    std::string message;
};

// Graph node ceiling of an unfixed library, S11-P6:
//   P_max = floor((max(n_ubatch*40, 32*n_tensors) - 1371) / 2156)
// Measured exactly at n_ubatch 128 -> 2, 256 -> 4, 512 -> 8.
// n_tensors 0 drops the second term, which lowers the ceiling and so refuses
// earlier. That is the safe direction.
int32_t seq_routing_pool_ceiling(uint32_t n_ubatch, int32_t n_tensors);

// Smallest --batch-size that admits a pool of n_pool on an unfixed library.
uint32_t seq_routing_batch_size_for(int32_t n_pool);

// W-4's pool bound, computed from device memory rather than from a constant.
//
// D3 capped the pool at 3 from S8's throughput decay. Stage 6A removed the
// decay (pool 8 is 1.01x pool 2 at one adapter in flight) and 08 removed the
// node ceiling, so the remaining bound is VRAM for the adapter weights.
//
// The pool is ALREADY RESIDENT when this runs: common_init_from_params loads
// every --lora before the routing config is built. So this is a headroom
// report, not a pre-flight check - a pool too large to load fails inside the
// loader, loudly, before any of this. What it answers is "how close to the edge
// is this server, and how many more adapters would fit".
struct seq_routing_vram_cap {
    bool    known             = false; // false = no device memory figure; W-4 falls back
    size_t  pool_bytes        = 0;     // adapter tensor data, summed from the GGUFs
    size_t  per_adapter_bytes = 0;     // mean over the pool, with the margin applied
    size_t  free_bytes        = 0;
    int32_t headroom          = 0;     // further adapters that fit in free_bytes
    int32_t cap               = 0;     // pool size + headroom
};

seq_routing_vram_cap seq_routing_pool_vram_cap(const seq_routing_config & cfg);

// Free and total memory of the first offload device, for W-4. Sets both to 0
// when there is no such device, which is the CPU-only case and not an error.
void seq_routing_device_memory(size_t * free_bytes, size_t * total_bytes);

// R-1 to R-7. An empty result means the server may start.
std::vector<seq_routing_finding> seq_routing_refusals(const seq_routing_config & cfg);

// W-1 to W-4.
std::vector<seq_routing_finding> seq_routing_warnings(const seq_routing_config & cfg);

// Read one adapter GGUF for R-5, R-7 and W-2. ptr may be null; when it is set,
// the alora flag comes from the loaded handle instead of the file.
seq_routing_adapter seq_routing_scan_adapter(const std::string & path, const llama_adapter_lora * ptr);

// Tensor count from a model GGUF header, for R-2. Returns 0 when the file
// cannot be read.
int32_t seq_routing_model_n_tensors(const std::string & path);

// Per-request refusals, both HTTP 400.
std::string seq_routing_q1_message(int32_t lora_id, size_t pool_size);
std::string seq_routing_q2_message(size_t n_enabled);
