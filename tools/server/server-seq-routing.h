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
