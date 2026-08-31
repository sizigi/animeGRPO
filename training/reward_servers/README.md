# Reward servers

Each scorer runs as its own FastAPI process holding one model, so the trainer
never has to co-locate a reward model with the policy. The reward functions are
thin HTTP clients (`training/reward_functions/`), which also means you can swap
any scorer for your own without touching the trainer.

Every server: `POST /score` and `GET /healthz` — **`/healthz`, not `/health`.**

| Server | Default port | Request | Response |
|---|---|---|---|
| `whisper_server.py` | 8001 | `{"tokens": [int], "text": str}` | `{"nll": float, "transcript": str}` |
| `animescore_server.py` | 8002 | `{"tokens": [int]}` | `{"raw_score": float}` |
| `likability_server.py` | 8003 | `{"tokens": [int]}` | `{"raw_score": float}` |
| `vad_server.py` | 8006 | `{"tokens": [int]}` | `{"arousal": float, "dominance": float, "valence": float}` |
| `utmos_server.py` | 8007 | `{"tokens": [int]}` | `{"utmos": float}` |

`tokens` are raw XCodec2 codec ids in `[128264, 193799]`. Every server decodes
them to a waveform with XCodec2 itself, so the trainer ships tokens over the
wire, not audio. CER is computed **client-side** by the reward function with
`jiwer` over the returned `transcript` — the `nll` field is available but not
what the paper's gate uses.

## Launching

```bash
REWARD_GPU=1 AXES="animescore" bash ../scripts/start_reward_servers.sh
```

Or one at a time:

```bash
CUDA_VISIBLE_DEVICES=1 python whisper_server.py    --port 8001 --model large-v3
CUDA_VISIBLE_DEVICES=1 python animescore_server.py --port 8002 --ckpt /path/to/animescore.pt
CUDA_VISIBLE_DEVICES=1 python utmos_server.py      --port 8007 --device cuda:0
```

The reward functions read `ANIMESCORE_SERVER`, `WHISPER_SERVER`,
`UTMOS_SERVER`, `LIKABILITY_SERVER`, `VAD_SERVER` from the environment
(defaulting to `http://localhost:<port>` above), so servers can live on another
host.

## Model weights

| Server | Model | Where it comes from |
|---|---|---|
| animescore | RankNet head + BiLSTM on frozen `microsoft/wavlm-base`, `--layer_mixing_k 4` | trained separately — see [sizigi/animescore](https://github.com/sizigi/animescore); pass with `--ckpt` |
| likability | 6-class head on `microsoft/wavlm-base-plus`, expectation over classes | trained separately; pass with `--ckpt` |
| whisper | `openai/whisper-large-v3` | downloaded by `openai-whisper` |
| utmos | UTMOS22-strong (`tarepan/SpeechMOS`) | downloaded on first run; CPU is fine |
| vad | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | downloaded from the Hub |

`animescore_server.py` and `likability_server.py` default `--ckpt` to the
authors' local paths. Pass your own; they are the only two that need weights
you must supply yourself.

## Failure behaviour

A reward function that cannot reach its scorer **raises** rather than silently
returning 0 — a dead server would otherwise train the policy on a constant
reward and look like a plateau. Health is re-checked at most every 30 s to keep
the check off the hot path. Whisper is the exception: a transcription failure is
treated as `CER = 1.0`, i.e. a constraint violation, since an utterance nothing
can transcribe should not earn style reward.
