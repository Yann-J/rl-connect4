from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch as th
from sb3_contrib import MaskablePPO


class MaskablePolicyOnnxWrapper(th.nn.Module):
    def __init__(self, policy: th.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor, action_masks: th.Tensor) -> th.Tensor:
        # Bypass MaskableCategorical construction to keep torch.export compatible.
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        return logits.masked_fill(~action_masks.bool(), -1e8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained MaskablePPO policy to ONNX for browser inference."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to a MaskablePPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--output",
        default="web/policy.onnx",
        type=str,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        default=18,
        type=int,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Registers custom extractor class path referenced in checkpoint metadata.
    from rl_connect4.policies.cnn_policy import (  # pylint: disable=import-error
        Connect4CNNExtractor,
    )

    model = MaskablePPO.load(
        str(model_path),
        custom_objects={"features_extractor_class": Connect4CNNExtractor},
    )
    wrapper = MaskablePolicyOnnxWrapper(model.policy).eval()

    obs_sample = th.zeros((1, 2, 6, 7), dtype=th.float32)
    mask_sample = th.ones((1, 7), dtype=th.bool)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch = th.export.Dim("batch")

    th.onnx.export(
        wrapper,
        (obs_sample, mask_sample),
        str(output_path),
        dynamo=True,
        external_data=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs", "action_masks"],
        output_names=["logits"],
        dynamic_shapes={
            "obs": {0: batch},
            "action_masks": {0: batch},
        },
    )
    print(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
