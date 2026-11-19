"""Headless 30s rollout plot of the push policy. No window.

Loads the push checkpoint the same way play_viewer.py does, pins the
viewer default command, and writes one PNG.
"""

import os

# A 4GB GPU cannot also give JAX most of its memory. Set this before JAX loads.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import play_viewer

PUSH_CHECKPOINT = (
    "/home/kingayush/King_stuff/RL_sim/logs/go1-push/"
    "Go1JoystickFlatTerrain-20261117-212556/checkpoints/000206438400"
)
PNG_PATH = "/home/kingayush/King_stuff/RL_sim/Sim_Videos/go1-push-rollout.png"
SECONDS = 30.0
VX, VY, YAW = 0.5, 0.0, 0.0


def make_control_step(env, policy):
  """One control step. Returns whether this step ended in a fall."""

  def control_step(state, key, command):
    key, act_key, reset_key = jax.random.split(key, 3)
    state = play_viewer._with_command(state, command)
    action, _ = policy(state.obs, act_key)
    stepped = env.step(state, action)
    fell = stepped.done > 0
    nxt = jax.lax.cond(
        fell,
        lambda _: env.reset(reset_key),
        lambda s: s,
        stepped,
    )
    nxt = play_viewer._with_command(nxt, command)
    return nxt, key, fell, stepped.data

  return jax.jit(control_step)


def rollout(env, control_step, command):
  """Step for SECONDS and keep one row per control step."""
  n = int(round(SECONDS / env.dt))
  rng = jax.random.PRNGKey(0)
  rng, reset_key = jax.random.split(rng)
  state = jax.jit(env.reset)(reset_key)
  state = play_viewer._with_command(state, command)
  rows = []
  falls = 0
  for i in range(n):
    state, rng, fell, data = control_step(state, rng, command)
    jax.block_until_ready(data.qpos)
    fell_f = float(fell)
    if fell_f > 0:
      falls += 1
    act = play_viewer._body_twist(env, data)
    kick = float(np.linalg.norm(play_viewer._kick_force(env, data)))
    rows.append((i * env.dt, VX, VY, YAW, act[0], act[1], act[2], kick, fell_f))
    if (i + 1) % max(1, int(round(5.0 / env.dt))) == 0:
      print(f"t={rows[-1][0]:.0f}s", flush=True)
  return np.asarray(rows, dtype=np.float64), falls


def save_plot(rows) -> None:
  t = rows[:, 0]
  fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
  pairs = (
      (1, 4, "vx (m/s)"),
      (2, 5, "vy (m/s)"),
      (3, 6, "yaw (rad/s)"),
  )
  for ax, (cmd_col, act_col, label) in zip(axes[:3], pairs):
    ax.plot(t, rows[:, cmd_col], color="0.35", linestyle="--", label="command")
    ax.plot(t, rows[:, act_col], color="C0", label="actual")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
  axes[0].legend(loc="upper right")
  axes[3].plot(t, rows[:, 7], color="C1")
  axes[3].set_ylabel("kick (N)")
  axes[3].set_xlabel("time (s)")
  axes[3].grid(True, alpha=0.3)
  fig.suptitle("Push policy: Go1 flat terrain, command vx 0.5, vy 0, yaw 0")
  fig.tight_layout()
  os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
  fig.savefig(PNG_PATH, dpi=140)
  plt.close(fig)


def main() -> None:
  path = play_viewer._resolve_checkpoint(PUSH_CHECKPOINT)
  print(f"Restoring from: {path}", flush=True)
  env = play_viewer.registry.load(
      play_viewer.ENV_NAME,
      config=play_viewer.registry.get_default_config(play_viewer.ENV_NAME),
      config_overrides={"impl": "jax"},
  )
  policy = play_viewer.load_policy(env, path)
  control_step = make_control_step(env, policy)
  command = jp.array([VX, VY, YAW], dtype=jp.float32)
  print("Compiling one reset and one step...", flush=True)
  rows, falls = rollout(env, control_step, command)
  save_plot(rows)
  kicks = int(np.count_nonzero(rows[:, 7] > 1.0))
  print(f"steps {len(rows)}  kick_steps {kicks}  falls {falls}", flush=True)
  print(PNG_PATH, flush=True)


if __name__ == "__main__":
  main()
