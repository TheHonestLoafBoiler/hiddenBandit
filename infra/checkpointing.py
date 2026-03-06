"""
Checkpointing
==============

Save and load agent state (policy weights, optimizer state, training step).
"""

import torch
from pathlib import Path


def save_checkpoint(agent, step: int, output_dir: str, final: bool = False):
    """Save agent state to disk.

    Args:
        agent:      GRPOAgent instance.
        step:       Current training step.
        output_dir: Directory to save checkpoint in.
        final:      If True, saves as 'checkpoint_final.pt' instead of numbered.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = "checkpoint_final.pt" if final else f"checkpoint_{step:06d}.pt"
    path = out_path / filename

    torch.save(
        {
            "step": step,
            "policy_state_dict": agent.policy.state_dict(),
            "ref_policy_state_dict": agent.ref_policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(agent, path: str):
    """Restore agent state from a checkpoint file.

    Args:
        agent: GRPOAgent instance (must have matching architecture).
        path:  Path to .pt checkpoint file.

    Returns:
        The training step at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.ref_policy.load_state_dict(checkpoint["ref_policy_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.train_step_count = checkpoint["step"]
    return checkpoint["step"]
