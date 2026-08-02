# P1 GetToastedBread MVP

이 checkout은 같은 Human300 EMA와 같은 locked `450/25/25` split으로 다음 세
실험을 실행한다.

| Experiment name | Config | 학습 대상 | Privileged input / gate |
|---|---|---|---|
| `P1-Baseline-DiT-Unfreeze` | `train_p1_baseline` | cached feature 조건의 기존 DiT만 | 없음 |
| `P1-Oracle-Gate-Symbol-DiT-Freeze` | `train_p1_oracle_gate_symbol` | P1 history/Flow/Jump/feedback만 | GT gate + window-start current symbol |
| `P1-Main-Full-DiT-Freeze` | `train_p1_main_full` | P1 history/Flow/Jump/feedback만 | train GT→predicted soft gate; eval predicted gate |

세 실험 모두 locked 969-D observation-feature cache와 cached action window를 읽는다. Baseline은 observation encoder를 freeze하고 DiT만 fine-tune하며, Oracle/Main은 observation encoder와 DiT를 모두 freeze한다. 같은
physical time의 overlap action에는 같은 Gaussian noise를 쓰고, 한 group의 네
window는 같은 diffusion timestep을 쓴다.

## 실행

저장소의 canonical simulation source를 `PYTHONPATH`에 둔 `sem_bif_dp` 환경에서
실행한다. MVP 세 실험의 고정 자원은 RTX 3090 24 GiB 6장이고, GPU당 3 episode를
처리한다. 이 값은 가장 무거운 Baseline의 FP32 profile과 세 실험의 공정한 global
batch를 함께 기준으로 정했다.

```console
export PYTHONPATH=/workspace/sem-bif-robot-traj/hub/srcs/diffusion_policy_p1_mvp:\
/workspace/sem-bif-robot-traj/hub/srcs/robomimic_robocasa:\
/workspace/sem-bif-robot-traj/hub/srcs/robosuite:\
/workspace/sem-bif-robot-traj/hub/srcs/robocasa365

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --multi_gpu --num_processes 6 --mixed_precision no --main_process_port 0 \
  diffusion_policy/workspace/train_p1_mvp_workspace.py \
  --config-name=train_p1_oracle_gate_symbol
```

다른 두 config도 `--config-name`만 바꾼다. 기본 W&B mode는 `offline`이고 local
scalar log는 run directory의 `logs.json.txt`, checkpoint는 `checkpoints/`에 저장된다.
`training.max_train_steps`는 smoke 전용이며 full run에서는 기본값 `null`을 유지한다.
짧은 smoke는 worker startup·shutdown 비용을 재는 실험이 아니므로
`dataloader.num_workers=0 val_dataloader.num_workers=0`도 함께 override한다. Full run은
아래 고정값인 train 4 / validation 2를 사용한다.

동일 run directory의 `checkpoints/latest.ckpt`에서 재개하려면 해당 directory를
고정하고 resume을 켠다.

```console
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
  --multi_gpu --num_processes 6 --mixed_precision no --main_process_port 0 \
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
- physical batch per rank: 3 episodes × 8 groups = 24 group instances
- group: `K=4`, `W=10`, stride `S=5`

Candidate pools are reconstructed once from semantic annotations and cached in
memory. The full 450-episode epoch audit yields 3,600 groups and 14,400 windows.
6-GPU에서는 한 global step이 18 episodes, 144 groups, 576 windows이고 정확히
25 step이 한 epoch이다. 450이 `6 × 3`으로 나누어지므로 epoch 끝 dummy가 없다.

## 6-GPU 고정 실행값

| 항목 | 값 |
|---|---|
| GPU | RTX 3090 24 GiB × 6 |
| Precision | FP32; MVP에서는 AMP를 사용하지 않음 |
| Local batch | 3 episodes = 24 groups = 96 DiT windows / rank |
| Global batch | 18 episodes = 144 groups = 576 DiT windows / optimizer step |
| Epoch | 25 optimizer steps; 450 real episodes, dummy 없음 |
| Train / validation workers | rank당 4 / 2 |
| Gradient accumulation | 1 |
| LR / warmup | `1e-4` / 125 steps (50-epoch 1,250-step budget의 10%) |
| Validation / checkpoint | 5 epoch마다 |

가장 무거운 Baseline의 single-GPU 실제 profile에서 3 episodes/rank는 peak reserved
17.35 GiB였다. 4 episodes/rank도 22.45 GiB로 한 번은 실행됐지만 DDP bucket,
allocator fragmentation과 운영 여유를 고려하면 안전한 full-run 값으로 쓰지 않는다.

## Trainable parameters

| 실험 | Trainable parameters | Breakdown |
|---|---:|---|
| P1-Baseline-DiT-Unfreeze | 63,575,052 | DiT 63,575,052 (observation encoder frozen) |
| P1-Oracle-Gate-Symbol-DiT-Freeze | 1,858,465 | history 367,584 + dynamics 1,350,849 + feedback 140,032 |
| P1-Main-Full-DiT-Freeze | 1,858,273 | history 367,392 + dynamics 1,350,849 + feedback 140,032 |

Baseline은 cached feature를 조건으로 DiT만 학습한다. Observation encoder에는 raw image가 입력되지 않으며 freeze된다. Oracle/Main도 같은 cache를 읽고 frozen DiT는 `no_grad`로 실행하며 약 186만 P1 parameter만 학습한다.
위 수는 각 production config로 실제 policy를 생성한 뒤 `requires_grad=True`인
parameter의 `numel()`을 합산한 값이다. Oracle과 Main의 192개 차이는
`6 symbols × 32 dimensions`인 oracle embedding이며 Main에서는 freeze된다.

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
reference path다. 세 mode 모두 6-GPU multi-step DDP smoke를 통과했지만 이 서버의 system
NVCC 11.7과 PyTorch CUDA 12.6이 달라 `mamba_ssm` fast extension은 아직 사용하지
않는다. 따라서 full pilot에서 reference backend 속도를 먼저 기록한다. Latent mode의 persistent online rollout은
아직 구현하지 않았고 `predict_action`은 vanilla DP로 조용히 fallback하지 않고
명시적으로 실패한다. Baseline rollout은 기존 `predict_action`을 그대로 사용한다.
