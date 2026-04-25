# RL Connect-4

This is my first important RL project to test my [shiny new RL certificate](https://digitalcredential.stanford.edu/check/B292F634C9E029D15DDEC09E41EEE4C0BD530FC1EBF9A0883544731EDE16BE72cy9ZQjNTYk50dTZRWWp0SDRWSlVLUjBJM2JYeTFlOVhvK1V3ZU5NRjBBbnNzMWVq), and was a long-standing ambition to build an AI agent to play Connect-4.

It includes both the ML project to train a model, as well as a simple web app that lets you play against the model online.

The online game is available [here](https://yann-j.github.io/rl-connect4).

![Screenshot](docs/screenshot.png)

This was developed from scratch but heavily inspired from the following references to get me started (though I ended up with very different choices):

- [Connect Zero](https://codebox.net/pages/connect4)
- [Clément Brutti-Mairesse](https://clementbm.github.io/project/2023/03/29/reinforcement-learning-connect-four-rllib.html)
- [AgileRL Tutorial](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/dqn.html)

## General architecture

- [PettingZoo](https://pettingzoo.farama.org/index.html) play environment (`connect_four_v3`)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) + sb3-contrib `MaskablePPO` RL framework
- PPO algorithm with legal action masking policy
- ML model using:
  - Residual CNN feature extractor (stack of 3x3 conv blocks)
  - Policy/value MLP heads on top of extracted features
- Simple sparse terminal reward (`-1/0/+1`), since games are quite short/finite
- TensorBoard metrics for reward and win-rate baselines

## Learning strategy

A lot of experimentation was done on opponent selection for training to create appropriate learning pressure (because just running the policy against itself could reward the policy for getting a better player, but also for being a worse opponent, making pure self-play learning potentially unstable):

- We are using a mix of random opponents with varied policies:
  - The current policy (pure self-play)
  - Some previous policy checkpoint
  - A random policy (mostly used initially to get the training off the ground)
  - Monte-Carlo Tree Search policy with increasing strength (simulating N games and picking best action)
  - A simple hardcoded rule-based policy (win if possible, block if needed, then expand or play random)
- The opponent mix gets increasingly harder over time, configured via timestep phases
- After every checkpoint we run a league/tournament between the last few checkpoints and pick the best for further training - this was a recommendation from various tutorials but in practice it doesn't look like it helped (we almost always select the latest).

## CI

The CI on this repo will run the training (while exposing a Tensorboard UI to watch learning metrics) and then deploy the webapp to Github Pages.

The Tensorboard logs from the CI runs are summarized in a static report also published in Github Pages [here](https://yann-j.github.io/rl-connect4/tensorboard).

Training takes several hours on the default Github runners (cpu), so CI is run on AWS GPUs managed via [RunsOn](https://runs-on.com/), which are much cheaper than GitHub GPUs.

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

Export the trained policy to ONNX (so it can be loaded in the browser):

```bash
python scripts/export_policy_onnx.py \
  --model checkpoints/<run_name>/final_model.zip \
  --output web/policy.onnx
```

Then serve the `web/` folder with any static file server:

```bash
python -m http.server 8000 --directory web
```

or:

```bash
npx serve -p 8000 web
```

Open `http://127.0.0.1:8000` and play directly in the browser (inference runs
client-side with `onnxruntime-web`).
