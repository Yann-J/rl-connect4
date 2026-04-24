# RL Connect-4

This is my first important RL project, a long-standing ambition to build an AI agent to play Connect-4.

It includes both an ML project to train a model, as well as a web app that lets you play against the model.

## General architecture

- PettingZoo play environment (`connect_four_v3`)
- Stable-Baselines3 + sb3-contrib `MaskablePPO` RL framework
- ML model using 2 CNN (because Connect-4 has strong geometric patterns based on neighbors) + 1 fully-connected linear layer
- Simple sparse rewards (`-1/0/+1`)
- Legal-action masking
- TensorBoard metrics for reward and win-rate baselines

## Learning strategy

A lot of tweaking was done on opponent selection for self-play training (because just running the policy against itself could reward the policy for getting a better player, but also for being a worse opponent):

- We are using a mix of random opponents with varied policies:
  - Some previous policy checkpoint
  - A random play policy
  - Monte-Carlo Tree Search policy (playing N random games and picking best action from the simulation)
  - A simple hardcoded rule-based policy (win if possible, block if needed, then expand or play random)
- The opponent selection gets increasingly harder over time, configured with timestep phases
- After every checkpoint we run a league/tournament between the last few checkpoints and pick the best for further training

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/train_self_play.py --config configs/train_self_play.yaml
```

## TensorBoard

```bash
tensorboard --logdir runs
```

## Fully static web game (ONNX in browser)

Export your trained policy to ONNX:

```bash
.venv/bin/python scripts/export_policy_onnx.py \
  --model checkpoints/<run_name>/final_model.zip \
  --output web/policy.onnx
```

Then serve the `web/` folder with any static file server:

```bash
python -m http.server 8000 --directory web
```

Open `http://127.0.0.1:8000` and play directly in the browser (inference runs
client-side with `onnxruntime-web`).
