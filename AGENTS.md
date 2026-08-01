# AGENTS.md — llama-infernal

Private, permanent fork of `ggml-org/llama.cpp` (MIT), owned by the
`inferference` project. This file supersedes upstream's `AGENTS.md`:
upstream's contributor guidelines target ggml-org PRs and do **not** apply
here (upstream itself: "Private forks are exempt").

Purpose: carry our modifications — the P2 mixed-batch multi-LoRA change and
the nanbeige arch — as **real commits** on `feat/*` branches off the ABI
anchor (`anchor/b10103` = `c588c4f47`), merged down to `integration`, shipped
as `release-<anchor>` tags.

## Environment

- **No repo-local dev environment.** The toolchain lives in the consuming
  repo: `inferference`'s `devenv.nix` provides the Python/CUDA/CMake/Ninja
  shell.
- All in-repo operations (build, smoke, bindgen, version control) run inside
  `devenv shell` from the `inferference` checkout, where this repo enters as a
  git submodule at `vendor/llama.cpp`.

## Version control

- One branch = one modification off the anchor (or off `integration` when the
  feature needs another feature). Never build on `master`.
- Feature work lands as **real commits with provenance**: the commit message
  records the source patch path and its `sha256` (keeps the manifest's
  `patch_sha256` derivable). No squashed patch blobs.
- From `inferference`, route version control through **gitman** (lanes =
  branches; `land` merges into trunk). The anchor never moves silently.
- Commit messages: plain, zero AI attribution.

## Verification

- Build + gates are driven from `inferference`, not from this tree:
  - `ci/build/build-llamacpp.sh` — profiles `cpu-light` / `cuda-3060` /
    `p2fork`; emits lib set + headers + `build-manifest.json`.
  - `ci/gates/abi_smoke_gate.py` — fork symbols resolve, Ornith GGUF loads +
    greedy ~32-token gen, tokenizer round-trip.
  - `ci/gates/cffi_bindgen.py` — compile the API-mode binding against THIS
    build's headers + lib. **Compile failure == ABI drift == STOP.**
- testee (pytest/ruff/ty) does not apply to this C++ tree; the fork's tests
  are the build + ABI gates above.

## Rules

- **Keep the fork surgical**: upstream tree + our commits only. No harness, no
  python, no benchmarks, no CI in this repo — that all lives in `inferference`.
- Never propose vLLM/SGLang as an answer.
- GPU smoke pins `CUDA_VISIBLE_DEVICES=0` (GPU 1 often hosts a vLLM runner).
- Fail-closed everywhere: a missing capability is a skip or a loud fail,
  never a silent pass.
