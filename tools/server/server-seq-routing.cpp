#include "server-seq-routing.h"

#include "common.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

// Tensor name fragments that build_lora_mm_id consumes (llama-graph.cpp call
// sites 2118, 2137, 2150, 2239; names llama-arch.cpp:401-404). That builder has
// no sequence-routing branch, so a delta on one of these is dropped in silence.
static const char * MOE_TENSOR_FRAGMENTS[] = {
    "ffn_gate_exps",
    "ffn_gate_up_exps",
    "ffn_down_exps",
    "ffn_up_exps",
};

// Graph nodes with an empty pool, and nodes per registered adapter, both
// measured (spikes/S8/pool-decay.md, S11-P6).
static const uint32_t SEQ_ROUTING_BASE_NODES        = 1371;
static const uint32_t SEQ_ROUTING_NODES_PER_ADAPTER = 2156;

static void * seq_routing_sym(const char * name) {
#ifdef _WIN32
    HMODULE h = GetModuleHandleA("llama.dll");
    return h ? (void *) GetProcAddress(h, name) : nullptr;
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

seq_routing_api seq_routing_api::resolve() {
    seq_routing_api api;

    void * a = seq_routing_sym("llama_set_seq_adapters");
    void * b = seq_routing_sym("llama_set_seq_adapter");
    void * c = seq_routing_sym("llama_set_seq_adapter_scaled");

    if (!a) { api.missing.push_back("llama_set_seq_adapters");       }
    if (!b) { api.missing.push_back("llama_set_seq_adapter");        }
    if (!c) { api.missing.push_back("llama_set_seq_adapter_scaled"); }

    memcpy(&api.set_seq_adapters,       &a, sizeof(a));
    memcpy(&api.set_seq_adapter,        &b, sizeof(b));
    memcpy(&api.set_seq_adapter_scaled, &c, sizeof(c));

    return api;
}

int32_t seq_routing_pool_ceiling(uint32_t n_ubatch, int32_t n_tensors) {
    uint32_t budget = n_ubatch * 40u;
    if (n_tensors > 0) {
        budget = std::max(budget, 32u * (uint32_t) n_tensors);
    }
    if (budget <= SEQ_ROUTING_BASE_NODES) {
        return 0;
    }
    return (int32_t) ((budget - SEQ_ROUTING_BASE_NODES) / SEQ_ROUTING_NODES_PER_ADAPTER);
}

uint32_t seq_routing_batch_size_for(int32_t n_pool) {
    if (n_pool < 0) {
        n_pool = 0;
    }
    const uint32_t needed = SEQ_ROUTING_BASE_NODES + SEQ_ROUTING_NODES_PER_ADAPTER * (uint32_t) n_pool;
    return (needed + 39u) / 40u;
}

int32_t seq_routing_model_n_tensors(const std::string & path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };

    gguf_context * gc = gguf_init_from_file(path.c_str(), gp);
    if (!gc) {
        return 0;
    }

    const int64_t n = gguf_get_n_tensors(gc);
    gguf_free(gc);

    return (int32_t) n;
}

seq_routing_adapter seq_routing_scan_adapter(const std::string & path, const llama_adapter_lora * ptr) {
    seq_routing_adapter out;
    out.path = path;

    if (ptr) {
        out.is_alora = llama_adapter_get_alora_n_invocation_tokens(ptr) > 0;
    }

    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };

    gguf_context * gc = gguf_init_from_file(path.c_str(), gp);
    if (!gc) {
        return out;
    }
    out.gguf_ok = true;

    const int64_t n = gguf_get_n_tensors(gc);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name = gguf_get_tensor_name(gc, i);

        if (name.size() > 7 && name.compare(name.size() - 7, 7, ".lora_a") == 0) {
            out.n_pairs++;
        }

        if (out.moe_tensor.empty()) {
            for (const char * frag : MOE_TENSOR_FRAGMENTS) {
                if (name.find(frag) != std::string::npos) {
                    out.moe_tensor = name;
                    break;
                }
            }
        }
    }

    gguf_free(gc);

    return out;
}

std::vector<seq_routing_finding> seq_routing_refusals(const seq_routing_config & cfg) {
    std::vector<seq_routing_finding> out;

    // R-1 [S9 direct resolution]
    if (!cfg.missing_symbols.empty()) {
        std::string names;
        for (const auto & n : cfg.missing_symbols) {
            names += names.empty() ? n : ", " + n;
        }
        out.push_back({ "R-1", string_format(
            "--lora-seq-routing: the loaded libllama does not export %s; this needs the P2 fork build "
            "(build_id in build-manifest.json: %s)",
            names.c_str(), cfg.build_info.c_str()) });

        // every later check needs the pool registered, so stop here
        return out;
    }

    // R-2 [S11-P6, stage 0.5 gate battery]. A library carrying the node-budget
    // fix has no ceiling: measured to pool 32 at n_batch 128, where the fix's
    // own parent aborts at pool 3. Do not fire there.
    if (!cfg.node_budget_fixed) {
        const int32_t p_max = seq_routing_pool_ceiling(cfg.n_ubatch, cfg.n_tensors);
        if ((int32_t) cfg.pool.size() > p_max) {
            out.push_back({ "R-2", string_format(
                "--lora-seq-routing: pool of %d adapters exceeds the graph node ceiling of %d at n_batch %u "
                "(model has %d tensors). This libllama does not budget the sequence adapter pool; past the ceiling "
                "the process aborts inside llama_decode. Raise --batch-size to %u, register at most %d adapters, "
                "or load a libllama carrying the stage 0.5 node-budget fix.",
                (int32_t) cfg.pool.size(), p_max, cfg.n_ubatch, cfg.n_tensors,
                seq_routing_batch_size_for((int32_t) cfg.pool.size()), p_max) });
        }
    }

    // R-3 [S11-P2]
    if (cfg.hats_registered) {
        out.push_back({ "R-3",
            "--lora-seq-routing cannot be combined with loop-adapter (hats) routing: both write the same adapter "
            "pool, and registering hats disables sequence routing entirely (verified bit-identical to the base "
            "model)." });
    }

    // R-4 [S11-P3, D6]. cache_idle_slots defaults to true, and the server turns
    // it off itself when the cache is off (server-context.cpp:1575-1578). It is
    // therefore live, and hazardous, only while cache_ram_mib is non-zero, so
    // the cache term alone covers it. --slot-save-path is independent.
    const bool cache_live = cfg.cache_ram_mib != 0;
    if (cache_live || !cfg.slot_save_path.empty()) {
        out.push_back({ "R-4",
            "--lora-seq-routing requires --cache-ram 0: the prompt cache is keyed on prompt tokens alone and "
            "carries no adapter identity, so a cached prompt computed under one adapter can be restored for a "
            "request naming another. Also incompatible: --cache-idle-slots, --slot-save-path." });
    }

    // R-5 [S11-P5]. An adapter whose GGUF cannot be read has unknown targets,
    // so it cannot be cleared. Refuse it rather than pass it: a scan that
    // cannot see a tensor name must never read as "no expert weights here".
    for (const auto & a : cfg.pool) {
        if (!a.gguf_ok) {
            out.push_back({ "R-5", string_format(
                "--lora-seq-routing: adapter %s could not be read as a GGUF, so the weights it targets are "
                "unknown. build_lora_mm_id has no sequence-routing branch, so a mixture-of-experts delta would "
                "be silently dropped. Refusing.",
                a.path.c_str()) });
        }
    }
    for (const auto & a : cfg.pool) {
        if (!a.moe_tensor.empty()) {
            out.push_back({ "R-5", string_format(
                "--lora-seq-routing: adapter %s targets mixture-of-experts weight %s. build_lora_mm_id has no "
                "sequence-routing branch, so this delta would be silently dropped. Refusing.",
                a.path.c_str(), a.moe_tensor.c_str()) });
        }
    }

    // R-6 [S11-P1]. Defence in depth: the stock server assigns
    // cparams.n_seq_max from params.n_parallel (common/common.cpp:1632), so
    // this is expected never to fire. It costs one comparison and it states the
    // invariant the routing path depends on.
    if (cfg.n_parallel > cfg.n_seq_max) {
        out.push_back({ "R-6", string_format(
            "--lora-seq-routing: --parallel %d exceeds the context's n_seq_max %d; a sequence id at or past "
            "n_seq_max reads seq_adapter_map out of bounds and may receive an adapter that was never requested.",
            cfg.n_parallel, cfg.n_seq_max) });
    }

    // R-7 [S2]
    for (const auto & a : cfg.pool) {
        if (a.is_alora) {
            out.push_back({ "R-7", string_format(
                "--lora-seq-routing does not support activated LoRA (alora) adapters: alora activation is "
                "per-slot state the sequence-routing path does not carry. Adapter: %s.",
                a.path.c_str()) });
        }
    }

    return out;
}

std::vector<seq_routing_finding> seq_routing_warnings(const seq_routing_config & cfg) {
    std::vector<seq_routing_finding> out;

    // W-1 [S6c, S10]
    if (cfg.n_parallel <= 8) {
        out.push_back({ "W-1", string_format(
            "--lora-seq-routing at --parallel %d: ggml-cuda selects the MMVQ matmul at batch width <= 8, where "
            "widening a decode costs 5.80 ms per added sequence. Measured end-to-end at c=8 with 154-pair "
            "adapters: 0.985x - a wash. Routing pays above 8.",
            cfg.n_parallel) });
    }

    // W-2 [S8, S10, S11-P5]. The pair count is the number that predicts the
    // cost, not the pool size.
    if (cfg.pool.size() > 1) {
        std::string pairs;
        for (const auto & a : cfg.pool) {
            const std::string k = a.gguf_ok ? std::to_string(a.n_pairs) : std::string("unknown");
            pairs += pairs.empty() ? k : ", " + k;
        }
        out.push_back({ "W-2", string_format(
            "--lora-seq-routing: pool of %d adapters carrying %s LoRA pairs. The unfused loop costs per "
            "REGISTERED adapter and scales with pair count: two 154-pair adapters cost 1.36x per decode step at "
            "c=16, two 3-pair adapters cost 1.02x.",
            (int32_t) cfg.pool.size(), pairs.c_str()) });
    }

    // W-3 [S11-P4]
    if (cfg.has_draft_model) {
        out.push_back({ "W-3",
            "--lora-seq-routing with speculative decoding: the draft context carries no adapters, so drafts come "
            "from the base model. Measured with synthetic probe adapters: 82 tokens drafted, 0 accepted. Verify "
            "your acceptance rate before keeping this on." });
    }

    // W-4 [S8]
    if (cfg.pool.size() > 3) {
        out.push_back({ "W-4", string_format(
            "--lora-seq-routing: pool of %d exceeds the documented budget of 3 (50%% of single-adapter "
            "throughput). At pool 8 routing is 1.82x slower per step than the stock serialization it replaces.",
            (int32_t) cfg.pool.size()) });
    }

    return out;
}

std::string seq_routing_q1_message(int32_t lora_id, size_t pool_size) {
    return string_format(
        "lora id %d is not in the routing pool (pool size %d); --lora-seq-routing fixes the pool at startup.",
        lora_id, (int32_t) pool_size);
}

std::string seq_routing_q2_message(size_t n_enabled) {
    return string_format(
        "--lora-seq-routing routes one sequence to one pool slot; this request enables %d adapters.",
        (int32_t) n_enabled);
}
