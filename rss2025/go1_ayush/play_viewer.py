"""Live viewer for the stock Go1 flat-terrain velocity policy.

The policy reads a 3-number command (forward, sideways, yaw rate) at the end
of its observation. This script pins that command to the CLI values, or to an
Xbox controller on /dev/input/js0 when one is connected, and steps one env in
a MuJoCo window. It does not write a video.
"""

import os

# A 4GB GPU cannot also give JAX most of its memory. Set this before JAX loads.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import functools
import math
import time

from brax.training import checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.locomotion.go1 import go1_constants

ENV_NAME = "Go1JoystickFlatTerrain"
LOGS_DIR = "/home/kingayush/King_stuff/RL_sim/logs"
# Finished stock run. Also the default when neither flag is set.
DEFAULT_CHECKPOINT = (
    LOGS_DIR + "/go1-stock/"
    "Go1JoystickFlatTerrain-20261111-221943/checkpoints/000206438400"
)
# Finished yaw-penalty run, weight -50.
YAW_CHECKPOINT = (
    LOGS_DIR + "/go1-yaw-penalty/"
    "Go1JoystickFlatTerrain-20261112-160732/checkpoints/000206438400"
)
JS_DEVICE = "/dev/input/js0"
# Resting Xbox sticks sit a little off zero. Inside this band the command is 0.
STICK_DEADZONE = 0.15


def _resolve_checkpoint(path: str) -> str:
  """Load a step dir, or the newest numbered step inside a checkpoints dir."""
  ckpt = os.path.abspath(path)
  if not os.path.isdir(ckpt):
    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")
  if os.path.exists(os.path.join(ckpt, "_CHECKPOINT_METADATA")):
    return ckpt
  numbered = []
  for name in os.listdir(ckpt):
    child = os.path.join(ckpt, name)
    if os.path.isdir(child) and name.isdigit():
      numbered.append(child)
  if not numbered:
    raise FileNotFoundError(
        f"No Brax checkpoint in {ckpt}. Pass the step directory, for example "
        f"{DEFAULT_CHECKPOINT}"
    )
  numbered.sort(key=lambda item: int(os.path.basename(item)))
  return numbered[-1]


def _newest_checkpoint(logs_dir: str) -> str:
  """Newest Brax step dir under logs_dir, by the metadata file's mtime."""
  found = []
  for dirpath, _dirnames, filenames in os.walk(logs_dir):
    if "_CHECKPOINT_METADATA" not in filenames:
      continue
    meta = os.path.join(dirpath, "_CHECKPOINT_METADATA")
    found.append((os.path.getmtime(meta), dirpath))
  if not found:
    raise FileNotFoundError(f"No Brax checkpoint under {logs_dir}")
  found.sort()
  return found[-1][1]


def checkpoint_for_args(args) -> str:
  """Pick the checkpoint. An explicit --checkpoint_path wins over --run."""
  if args.checkpoint_path is not None:
    return args.checkpoint_path
  if args.run is None or args.run == "stock":
    return DEFAULT_CHECKPOINT
  if args.run == "yaw":
    return YAW_CHECKPOINT
  if args.run == "latest":
    return _newest_checkpoint(LOGS_DIR)
  raise ValueError(f"Unknown --run value: {args.run}")


def _with_command(state, command):
  """Write the fixed command into info and into both observation vectors."""
  info = dict(state.info)
  info["command"] = command
  obs = dict(state.obs)
  state_obs = obs["state"].at[-3:].set(command)
  obs["state"] = state_obs
  # privileged_state starts with the same 48-d policy observation.
  obs["privileged_state"] = obs["privileged_state"].at[: state_obs.shape[0]].set(
      state_obs
  )
  return state.replace(info=info, obs=obs)


def load_policy(env, checkpoint_path: str):
  """Restore the policy the way train_jax_ppo.py --play_only does.

  play_only calls Brax PPO with num_timesteps=0. That builds this same network,
  loads the checkpoint tuple (normalizer, policy, value), and returns a
  deterministic inference function. This skips the training allocations.
  """
  ppo_params = locomotion_params.brax_ppo_config(ENV_NAME, impl="jax")
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory,
      preprocess_observations_fn=running_statistics.normalize,
  )
  ppo_network = network_factory(env.observation_size, env.action_size)
  params = checkpoint.load(checkpoint_path)
  params = jax.tree.map(jp.asarray, params)
  inference_fn = ppo_networks.make_inference_fn(ppo_network)(
      params, deterministic=True
  )
  return jax.jit(inference_fn)


def make_control_step(env, policy):
  """One control step: fixed command, policy action, reset if the robot falls."""

  def control_step(state, key, command):
    key, act_key, reset_key = jax.random.split(key, 3)
    state = _with_command(state, command)
    action, _ = policy(state.obs, act_key)
    state = env.step(state, action)
    state = jax.lax.cond(
        state.done > 0,
        lambda _: env.reset(reset_key),
        lambda stepped: stepped,
        state,
    )
    state = _with_command(state, command)
    return state, key

  return jax.jit(control_step)


def sync_mujoco(model, data, state) -> None:
  """Copy the MJX state into the viewer model. Do not step MuJoCo again."""
  mjx_data = state.data
  data.qpos[:] = np.asarray(mjx_data.qpos)
  data.qvel[:] = np.asarray(mjx_data.qvel)
  data.ctrl[:] = np.asarray(mjx_data.ctrl)
  data.time = float(mjx_data.time)
  if model.nmocap:
    data.mocap_pos[:] = np.asarray(mjx_data.mocap_pos)
    data.mocap_quat[:] = np.asarray(mjx_data.mocap_quat)
  data.xfrc_applied[:] = np.asarray(mjx_data.xfrc_applied)
  mujoco.mj_forward(model, data)


def prepare(checkpoint_path: str, vx: float, vy: float, yaw: float):
  """Load the flat-terrain env and compile one reset plus one control step."""
  checkpoint_path = _resolve_checkpoint(checkpoint_path)
  print(f"Restoring from: {checkpoint_path}")
  # The file default is warp. The stock run overrode this to jax.
  env = registry.load(
      ENV_NAME,
      config=registry.get_default_config(ENV_NAME),
      config_overrides={"impl": "jax"},
  )
  policy = load_policy(env, checkpoint_path)
  control_step = make_control_step(env, policy)
  command = jp.array([vx, vy, yaw], dtype=jp.float32)
  print(f"Command vx={vx} vy={vy} yaw={yaw}")
  print("Compiling the policy and one env step...")
  rng = jax.random.PRNGKey(0)
  rng, reset_key = jax.random.split(rng)
  state = jax.jit(env.reset)(reset_key)
  state, rng = control_step(state, rng, command)
  jax.block_until_ready(state.data.qpos)
  state_size = int(state.obs["state"].shape[0])
  if state_size != 48 or int(state.obs["privileged_state"].shape[0]) != 123:
    raise RuntimeError(
        "Observation size does not match the stock checkpoint "
        f"(state {state.obs['state'].shape}, privileged "
        f"{state.obs['privileged_state'].shape})."
    )
  print(
      "Compiled. Torso height"
      f" {float(state.data.qpos[2]):.3f} m. Close the window to stop."
  )
  return env, control_step, state, rng, command


def rollout(control_step, state, rng, command, steps: int):
  """Step without a window. Used to check the policy when no display is available."""
  height_sum = 0.0
  for _ in range(steps):
    state, rng = control_step(state, rng, command)
    height_sum += float(state.data.qpos[2])
  jax.block_until_ready(state.data.qpos)
  return state, rng, height_sum / steps


def _apply_deadzone(value: float, deadzone: float = STICK_DEADZONE) -> float:
  """Zero a resting stick. Stretch the rest so a full push is still ±1."""
  magnitude = abs(value)
  if magnitude < deadzone:
    return 0.0
  stretched = (magnitude - deadzone) / (1.0 - deadzone)
  return math.copysign(min(stretched, 1.0), value)


def _init_pygame():
  """Open pygame's joystick module without taking the MuJoCo window."""
  os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
  os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
  # This Xbox pad is the kernel js device. Skip SDL's HIDAPI copy of it.
  os.environ["SDL_JOYSTICK_HIDAPI"] = "0"
  os.environ["SDL_JOYSTICK_HIDAPI_XBOX"] = "0"
  os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
  import pygame

  pygame.init()
  pygame.joystick.init()
  return pygame


def open_controller():
  """Return the pygame joystick for /dev/input/js0, or None to keep the flags."""
  if not os.path.exists(JS_DEVICE):
    print(
        f"No controller connected at {JS_DEVICE}. "
        "Using the --vx --vy --yaw flags."
    )
    return None
  pygame = _init_pygame()
  pygame.event.pump()
  if pygame.joystick.get_count() < 1:
    print(
        f"No controller connected at {JS_DEVICE}. "
        "Using the --vx --vy --yaw flags."
    )
    return None
  stick = pygame.joystick.Joystick(0)
  stick.init()
  # Kernel js map for this pad: 0 left X, 1 left Y, 2 right X, 3 right Y.
  # Axes 4 and 5 are the triggers and rest at -1. Yaw uses axis 2.
  if stick.get_numaxes() < 3:
    print(
        f"Controller {stick.get_name()} at {JS_DEVICE} has "
        f"{stick.get_numaxes()} axes. Using the --vx --vy --yaw flags."
    )
    return None
  print(
      f"Controller connected ({stick.get_name()}) at {JS_DEVICE}. "
      "Sticks override --vx --vy --yaw."
  )
  return stick


def command_from_sticks(stick, limits):
  """Map the Xbox sticks onto the command ranges joystick.py samples.

  limits is command_config.a: [vx, vy, yaw] = [1.5, 0.8, 1.2] on the stock env.
  Stick up and stick left are the positive directions.

  Returns None when every stick axis is exactly -1. This pad reports that
  when it is connected but not sending a real stick position (asleep). The
  caller keeps the flag command until a stick moves.
  """
  import pygame

  pygame.event.pump()
  left_x = stick.get_axis(0)
  left_y = stick.get_axis(1)
  right_x = stick.get_axis(2)
  right_y = stick.get_axis(3) if stick.get_numaxes() > 3 else 0.0
  asleep = all(abs(axis + 1.0) < 1e-3 for axis in (left_x, left_y, right_x, right_y))
  if asleep:
    if not getattr(command_from_sticks, "asleep_warned", False):
      print(
          "Sticks are reporting -1 on every axis (pad asleep, not a real"
          " push). Using --vx --vy --yaw until a stick moves."
      )
      command_from_sticks.asleep_warned = True
    return None
  if getattr(command_from_sticks, "asleep_warned", False):
    print("Controller is sending stick positions. Sticks override --vx --vy --yaw.")
    command_from_sticks.asleep_warned = False
  vx = -_apply_deadzone(left_y) * float(limits[0])
  vy = -_apply_deadzone(left_x) * float(limits[1])
  yaw = -_apply_deadzone(right_x) * float(limits[2])
  return jp.array([vx, vy, yaw], dtype=jp.float32)


def _body_twist(env, data):
  """Body-frame forward speed, sideways speed, and yaw rate.

  These are the readings the tracking reward compares to the command:
  local linear velocity x/y, and the z component of the gyro.
  """
  lin = np.asarray(env.get_local_linvel(data))
  gyro = np.asarray(env.get_gyro(data))
  return (float(lin[0]), float(lin[1]), float(gyro[2]))


def _fmt_twist(label, values):
  vx, vy, yaw = values
  return f"{label} vx={vx: .2f} vy={vy: .2f} yaw={yaw: .2f}"


def _kick_force(env, data):
  """World-frame force on the trunk, in newtons.

  This is the vector `_maybe_apply_perturbation` writes into
  `xfrc_applied` for this step: `force * pert_dir`. The wait branch
  writes zeros, so a quiet step is 0 N. The kick code is not changed.
  """
  return np.asarray(data.xfrc_applied[env._torso_body_id, :3], dtype=np.float64)


def _fmt_kick(force) -> str:
  """`kick 0.0 N`, or the newton magnitude and the unit direction."""
  mag = float(np.linalg.norm(force))
  if mag < 1e-3:
    return "kick 0.0 N"
  direction = force / mag
  return (
      f"kick {mag:.1f} N"
      f" dir x={direction[0]: .2f} y={direction[1]: .2f} z={direction[2]: .2f}"
  )


def _draw_kick(viewer, origin, force) -> None:
  """Draw the kick into the passive viewer's user scene.

  MuJoCo 3.16 copies `user_scn` geoms during `sync` and appends them
  after the robot scene. The geom is finished first, then `ngeom` is
  set, so sync cannot publish a cleared arrow. The shaft starts just
  above the trunk, in the world frame, and is at least 0.5 m long so
  it clears the body.
  """
  scn = viewer.user_scn
  if scn is None:
    raise RuntimeError(
        "This MuJoCo viewer has no user scene, so it cannot draw the kick arrow."
    )
  mag = float(np.linalg.norm(force))
  if mag < 1e-3:
    scn.ngeom = 0
    return
  direction = np.asarray(force, dtype=np.float64) / mag
  # 0.5 m clears the trunk. Extra length tracks the force, up to 1.6 m.
  length = float(np.clip(0.5 + mag / 120.0, 0.5, 1.6))
  start = np.asarray(origin, dtype=np.float64).reshape(3).copy()
  start[2] += 0.08
  end = start + direction * length
  geom = scn.geoms[0]
  mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW,
      np.zeros(3, dtype=np.float64),
      np.zeros(3, dtype=np.float64),
      np.zeros(9, dtype=np.float64),
      np.array([1.0, 0.35, 0.0, 1.0], dtype=np.float32),
  )
  # Width is the shaft radius in metres. 4.5 cm reads at the default camera.
  mujoco.mjv_connector(
      geom,
      int(mujoco.mjtGeom.mjGEOM_ARROW),
      0.045,
      start,
      end,
  )
  geom.category = mujoco.mjtCatBit.mjCAT_DECOR
  geom.emission = 1.0
  geom.specular = 0.3
  scn.ngeom = 1


def _print_twist(prefix, command, actual, kick):
  error = tuple(command[i] - actual[i] for i in range(3))
  print(
      f"{prefix}  {_fmt_twist('cmd', command)}"
      f"  {_fmt_twist('act', actual)}"
      f"  {_fmt_twist('err', error)}"
      f"  {_fmt_kick(kick)}",
      flush=True,
  )


def view(env, control_step, state, rng, command, stick=None) -> None:
  """Open a passive viewer and step in real time at the env control rate."""
  model = env.mj_model
  data = mujoco.MjData(model)
  sync_mujoco(model, data, state)
  limits = env._config.command_config.a
  print_every = max(1, int(round(1.0 / env.dt)))
  steps = 0
  sum_cmd = [0.0, 0.0, 0.0]
  sum_act = [0.0, 0.0, 0.0]
  sum_err = [0.0, 0.0, 0.0]
  # MuJoCo 3.16: these set simulate.ui0_enable and ui1_enable before the window draws.
  # Tab toggles the left panel. Shift-Tab toggles the right panel.
  with mujoco.viewer.launch_passive(
      model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(go1_constants.ROOT_BODY).id
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20.0
    viewer.cam.azimuth = 90.0
    print("Opening viewer.")
    # Commands are applied from the first control step. There is no
    # separate standup. Say so once, when the loop starts taking vx/vy/yaw.
    print("Accepting commands.", flush=True)
    while viewer.is_running():
      tick = time.perf_counter()
      if stick is not None:
        try:
          live = command_from_sticks(stick, limits)
        except Exception as exc:  # stick unplugged mid-run
          print(
              f"Controller read failed ({exc}). "
              "Using the --vx --vy --yaw flags."
          )
          stick = None
          live = None
        if live is not None:
          command = live
      state, rng = control_step(state, rng, command)
      jax.block_until_ready(state.data.qpos)
      cmd = tuple(float(x) for x in np.asarray(command))
      act = _body_twist(env, state.data)
      for i in range(3):
        sum_cmd[i] += cmd[i]
        sum_act[i] += act[i]
        sum_err[i] += cmd[i] - act[i]
      steps += 1
      kick = _kick_force(env, state.data)
      # The pulse lasts 0.05–0.2 s, so a 1 s sample misses it. While the
      # force is on, print every control step. Quiet steps stay on the 1 s line.
      if float(np.linalg.norm(kick)) >= 1e-3:
        print(
            f"t={steps * env.dt:.2f}s  {_fmt_kick(kick)}",
            flush=True,
        )
      if steps % print_every == 0:
        _print_twist(f"t={steps * env.dt:.0f}s", cmd, act, kick)
      sync_mujoco(model, data, state)
      _draw_kick(viewer, data.xpos[env._torso_body_id], kick)
      viewer.sync()
      remain = env.dt - (time.perf_counter() - tick)
      if remain > 0:
        time.sleep(remain)
  if steps == 0:
    print("No control steps, no velocity mean.")
    return
  seconds = steps * env.dt
  mean_cmd = tuple(value / steps for value in sum_cmd)
  mean_act = tuple(value / steps for value in sum_act)
  mean_err = tuple(value / steps for value in sum_err)
  print(
      f"mean over {seconds:.1f}s ({steps} steps)"
      f"  {_fmt_twist('cmd', mean_cmd)}"
      f"  {_fmt_twist('act', mean_act)}"
      f"  {_fmt_twist('err', mean_err)}",
      flush=True,
  )


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Watch the stock Go1 flat-terrain policy in a live window."
  )
  parser.add_argument(
      "--run",
      choices=("stock", "yaw", "latest"),
      default=None,
      help=(
          "stock: finished stock run. yaw: finished yaw-penalty run. "
          "latest: newest checkpoint under the logs folder. "
          "Ignored when --checkpoint_path is set."
      ),
  )
  parser.add_argument(
      "--checkpoint_path",
      default=None,
      help=(
          "Brax PPO checkpoint. Overrides --run. "
          "A step directory, or the checkpoints folder. "
          "With neither flag, the stock run is used."
      ),
  )
  parser.add_argument("--vx", type=float, default=0.5, help="Forward speed, m/s.")
  parser.add_argument("--vy", type=float, default=0.0, help="Sideways speed, m/s.")
  parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate, rad/s.")
  return parser.parse_args(argv)


def main(argv=None) -> None:
  args = parse_args(argv)
  stick = open_controller()
  env, control_step, state, rng, command = prepare(
      checkpoint_for_args(args), args.vx, args.vy, args.yaw
  )
  view(env, control_step, state, rng, command, stick)
  # GLFW asserts in _glfwPlatformSetTls while Python is shutting down.
  # The window has already closed. Skip that destructor.
  os._exit(0)


if __name__ == "__main__":
  main()
