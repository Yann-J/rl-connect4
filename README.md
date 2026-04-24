# RL Connect-4

This is my first important RL project, a long-standing ambition to build an AI agent to play Connect-4.

It includes both an ML project to train a model, as well as a web app that lets you play against the model.

This was developed from scratch but heavily inspired from the following references to get me started (though I ended up with very different choices):

- [Connect Zero](https://codebox.net/pages/connect4)
- [Clément Brutti-Mairesse](https://clementbm.github.io/project/2023/03/29/reinforcement-learning-connect-four-rllib.html)
- [AgileRL Tutorial](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/dqn.html)

## General architecture

- [PettingZoo](https://pettingzoo.farama.org/index.html) play environment (`connect_four_v3`)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) + sb3-contrib `MaskablePPO` RL framework
- PPO algorithm with legal-action masking policy
- ML model using:
  - Residual CNN feature extractor (stack of 3x3 conv blocks)
  - Policy/value MLP heads on top of extracted features
- Simple sparse rewards (`-1/0/+1`), since games are quite short/finite
- TensorBoard metrics for reward and win-rate baselines

## Learning strategy

A lot of experimentation was done on opponent selection for self-play training (because just running the policy against itself could reward the policy for getting a better player, but also for being a worse opponent):

- We are using a mix of random opponents with varied policies:
  - The current policy (pure self-play)
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
python scripts/train.py --config configs/train.yaml
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

```bash
npx serve -p 8000 web
```

Open `http://127.0.0.1:8000` and play directly in the browser (inference runs
client-side with `onnxruntime-web`).
