# P1 GetToastedBread MVP

이 checkout은 같은 Human300 EMA와 같은 locked `450/25/25` split으로 다음 세
실험을 실행한다.

| Config | 학습 대상 | Privileged input / gate |
|---|---|---|
| `train_p1_baseline` | observation encoder + 기존 DiT 전체 | 없음 |
| `train_p1_oracle_gate_symbol` | P1 history/Flow/Jump/feedback만 | GT gate + window-start current symbol |
| `train_p1_main_full` | P1 history/Flow/Jump/feedback만 | train GT→predicted soft gate; eval predicted gate |

세 실험 모두 raw RGB/proprioception/language를 읽는다. Oracle/Main의 pretrained
observation encoder와 DiT는 parameter뿐 아니라 train/eval mode도 freeze한다. 같은
physical time의 overlap action에는 같은 Gaussian noise를 쓰고, 한 group의 네
window는 같은 diffusion timestep을 쓴다.

## 실행

저장소의 canonical simulation source를 `PYTHONPATH`에 둔 `sem_bif_dp` 환경에서
실행한다.

```console
accelerate launch --num_processes 8 \
  diffusion_policy/workspace/train_p1_mvp_workspace.py \
  --config-name=train_p1_oracle_gate_symbol
```

다른 두 config도 `--config-name`만 바꾼다. 기본 W&B mode는 `offline`이고 local
scalar log는 run directory의 `logs.json.txt`, checkpoint는 `checkpoints/`에 저장된다.

동일 run directory의 `checkpoints/latest.ckpt`에서 재개하려면 해당 directory를
고정하고 resume을 켠다.

```console
accelerate launch --num_processes 8 \
  diffusion_policy/workspace/train_p1_mvp_workspace.py \
  --config-name=train_p1_main_full \
  training.resume=true \
  hydra.run.dir=/absolute/path/to/the/original/run
```

Checkpoint는 model, EMA model, optimizer, scheduler, epoch, mid-epoch batch cursor,
episodes seen, EMA decay step을 보존한다. Sampler와 diffusion/crop randomness는
`(seed, epoch, batch, rank)`로 다시 생성한다.
현재 exact-resume contract는 `gradient_accumulate_every=1`을 요구하며, 다른 값은
workspace 시작 시 오류로 거부한다.

## Data contract

- split manifest:
  `/data/sem_bif/derived/p1_composite_seen_split/split_manifest.json`
- required manifest SHA-256:
  `1ead92ce1b88d0fdd038b57b56b33a16be482d871099a32aad6360fe97ee6f53`
- task/cache intersection: train 450, validation 25, test 25
- physical batch per rank: 4 episodes × 8 groups = 32 group instances
- group: `K=4`, `W=10`, stride `S=5`

Candidate pools are reconstructed once from semantic annotations and cached in
memory. The full 450-episode epoch audit yields 3,600 groups and 14,400 windows.

## 검증과 현재 제한

```console
python -m pytest -q \
  tests/test_p1_workspace.py \
  tests/test_p1_mvp_group_dataset.py \
  tests/test_p1_latent_model.py \
  tests/test_p1_policy.py \
  tests/test_p1_mvp_sampler.py \
  tests/test_transformer_split_forward.py
```

현재 production 기본 history backend는 Transformers의 exact sequential Mamba
reference path다. CUDA fast kernel이 없는 환경에서도 의미는 정확하지만, full run
전에 3090 memory/speed profile이 필요하다. Latent mode의 persistent online rollout은
아직 구현하지 않았고 `predict_action`은 vanilla DP로 조용히 fallback하지 않고
명시적으로 실패한다. Baseline rollout은 기존 `predict_action`을 그대로 사용한다.
