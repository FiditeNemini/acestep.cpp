#pragma once
// pipeline-synth-impl.h: private definition of AceSynth (opaque pipeline handle)
//
// AceSynth holds all loaded models and derived state for the synthesis pipeline.
// The public API (pipeline-synth.h) forward-declares it as an opaque pointer.
// This header is included by the two implementation files:
//   pipeline-synth.cpp      orchestrator: load, generate, free
//   pipeline-synth-ops.cpp  primitives: encode, context, noise, dit, vae

#include "bpe.h"
#include "cond-enc.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "pipeline-synth.h"
#include "qwen3-enc.h"
#include "vae.h"

#include <vector>

struct AceSynth {
    // Models (loaded once)
    DiTGGML      dit;
    Qwen3GGML    text_enc;
    CondGGML     cond_enc;
    VAEGGML      vae;
    DetokGGML    detok;
    TokGGML      tok;
    BPETokenizer bpe;

    // Metadata from DiT GGUF
    bool               is_turbo;
    std::vector<float> silence_full;  // [15000, 64] f32

    // Config
    AceSynthParams params;
    bool           have_vae;
    bool           have_detok;
    bool           have_tok;

    // Derived constants
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)
};
