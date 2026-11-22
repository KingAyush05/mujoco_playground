# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joystick task for Go1."""

import copy
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

# One checker floor, 12 m by 10 m. The robot stands at the origin facing +x.
# Stairs are ahead, the slope is to the left, and the bump patch is to the right.
_FLOOR_RX = 6.0
_FLOOR_RY = 5.0
_FLOOR_RES = 0.05
_FLOOR_NCOL = int(round(2.0 * _FLOOR_RX / _FLOOR_RES)) + 1
_FLOOR_NROW = int(round(2.0 * _FLOOR_RY / _FLOOR_RES)) + 1
_TREAD_M = 0.2
_RISE_M = 0.02
_N_STEPS = 6
_STAIR_WIDTH = 1.0
_STAIR_X0 = 1.5
_SLOPE_DEG = 5.0
_SLOPE_LEN = 2.0
_SLOPE_WIDTH = 1.0
_SLOPE_X0 = -1.0
_SLOPE_Y0 = 1.5
_BUMP_SIZE = 1.0
_BUMP_CX = 0.0
_BUMP_CY = -2.0
_BUMP_MAX_M = 0.05
_SLOPE_ANGLES_DEG = (5.0, 10.0, 15.0, 20.0, 25.0)
_TRACK_SUCCESS = 0.6
_TRACK_FAIL = 0.35
_N_LEVELS = len(_SLOPE_ANGLES_DEG)
# Kept so an old caller can still pass a name. Placement is always the origin.
_PINNED_TERRAIN_TYPE = {
    "flat": 0,
    "bumps": 1,
    "slope": 2,
    "stairs": 3,
}


def _build_terrain_mix(mj_model: mujoco.MjModel) -> Dict[str, np.ndarray]:
  """Fill one small floor: a stair flight, a 5° slope, and a 5 cm bump patch."""
  nrow = int(mj_model.hfield_nrow[0])
  ncol = int(mj_model.hfield_ncol[0])
  if nrow != _FLOOR_NROW or ncol != _FLOOR_NCOL:
    raise RuntimeError(
        f"Mix hfield is {nrow}x{ncol}, expected {_FLOOR_NROW}x{_FLOOR_NCOL}."
    )

  rough = mujoco.MjModel.from_xml_string(
      consts.FEET_ONLY_ROUGH_TERRAIN_XML.read_text(),
      assets=go1_base.get_assets(),
  )
  bump = np.asarray(rough.hfield_data, dtype=np.float32).reshape(
      int(rough.hfield_nrow[0]), int(rough.hfield_ncol[0])
  )
  bump_m = bump * float(rough.hfield_size[0, 2])

  xs = np.linspace(-_FLOOR_RX, _FLOOR_RX, ncol, dtype=np.float32)
  ys = np.linspace(-_FLOOR_RY, _FLOOR_RY, nrow, dtype=np.float32)
  x_grid, y_grid = np.meshgrid(xs, ys)
  height = np.zeros((nrow, ncol), dtype=np.float32)

  stair_x1 = _STAIR_X0 + _N_STEPS * _TREAD_M
  stair = (
      (x_grid >= _STAIR_X0)
      & (x_grid < stair_x1)
      & (np.abs(y_grid) <= _STAIR_WIDTH / 2.0)
  )
  step = np.floor((x_grid - _STAIR_X0) / _TREAD_M)
  height[stair] = (step[stair] + 1.0) * _RISE_M

  slope = (
      (x_grid >= _SLOPE_X0)
      & (x_grid <= _SLOPE_X0 + _SLOPE_LEN)
      & (y_grid >= _SLOPE_Y0)
      & (y_grid <= _SLOPE_Y0 + _SLOPE_WIDTH)
  )
  height[slope] = (x_grid[slope] - _SLOPE_X0) * np.tan(np.deg2rad(_SLOPE_DEG))

  bump_mask = (
      (np.abs(x_grid - _BUMP_CX) <= _BUMP_SIZE / 2.0)
      & (np.abs(y_grid - _BUMP_CY) <= _BUMP_SIZE / 2.0)
  )
  u = (x_grid[bump_mask] - (_BUMP_CX - _BUMP_SIZE / 2.0)) / _BUMP_SIZE
  v = (y_grid[bump_mask] - (_BUMP_CY - _BUMP_SIZE / 2.0)) / _BUMP_SIZE
  bump_col = np.clip(
      np.rint(u * (bump.shape[1] - 1)).astype(np.int32), 0, bump.shape[1] - 1
  )
  bump_row = np.clip(
      np.rint(v * (bump.shape[0] - 1)).astype(np.int32), 0, bump.shape[0] - 1
  )
  height[bump_mask] = np.clip(bump_m[bump_row, bump_col], 0.0, _BUMP_MAX_M)

  elev = float(mj_model.hfield_size[0, 2])
  if elev <= 0.0:
    raise RuntimeError("Mix hfield elevation must be positive.")
  if float(height.max()) > elev:
    raise RuntimeError(
        f"Terrain height {float(height.max()):.3f} m exceeds the hfield."
    )
  mj_model.hfield_data[:] = height.reshape(-1) / elev
  return {"height_m": height}


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              # Ayush: yaw-rate penalty, only when the yaw command is ~0.
              # Weight -50, raised from -0.05 so the penalty still matters
              # after the reward sum is clipped at 0.
              yaw_rate=-50,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.5,
              # Other.
              termination=-1.0,
              stand_still=-1.0,
              # Regularization.
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              # Feet.
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.1,
      ),
      pert_config=config_dict.create(
          # Ayush: push recovery. This kick was already in the file, only off.
          # Direction is horizontal. One speed and one gap are drawn per
          # episode from the ranges below. Not trained yet.
          enable=True,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.5, 0.8, 1.2],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="warp",
      naconmax=4 * 8192,
      njmax=40,
  )


class Joystick(go1_base.Go1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    self._terrain_mix = task == "terrain_mix"
    # Brax caches the first pose unless this is set. The wrapper reads it.
    self.needs_full_reset = self._terrain_mix
    if task.startswith("rough"):
      config.naconmax = 8 * 8192
      config.njmax = 12 + 48
    elif self._terrain_mix:
      # Copy so the shared flat/rough default keeps its contact limits
      # and its push kicks.
      config = copy.deepcopy(config)
      config.naconmax = 8 * 8192
      config.njmax = 12 + 48
      config.pert_config.enable = False
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _prepare_mj_model(self, mj_model: mujoco.MjModel) -> None:
    if not self._terrain_mix:
      return
    built = _build_terrain_mix(mj_model)
    self._terrain_m = jp.asarray(built["height_m"])
    self._mix_ncol = int(built["height_m"].shape[1])
    self._mix_nrow = int(built["height_m"].shape[0])

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
    # Torso height of the home keyframe. On the mix field this is the
    # clearance added on top of the tile height.
    self._spawn_z = self._init_q[2] + 0.01

  def _terrain_z(self, x: jax.Array, y: jax.Array) -> jax.Array:
    col = (x + _FLOOR_RX) / (2.0 * _FLOOR_RX) * (self._mix_ncol - 1)
    row = (y + _FLOOR_RY) / (2.0 * _FLOOR_RY) * (self._mix_nrow - 1)
    col_i = jp.clip(jp.rint(col), 0, self._mix_ncol - 1).astype(jp.int32)
    row_i = jp.clip(jp.rint(row), 0, self._mix_nrow - 1).astype(jp.int32)
    return self._terrain_m[row_i, col_i]

  def place_terrain_mix(
      self,
      state: mjx_env.State,
      level: jax.Array,
      rng: jax.Array,
      pinned: Optional[str] = None,
  ) -> mjx_env.State:
    """Stand on the open floor at the origin, facing +x. One map, no tile pick."""
    del level, pinned
    xy = jp.zeros((2,))
    z = self._terrain_z(xy[0], xy[1]) + self._spawn_z
    qpos = state.data.qpos.at[0:3].set(jp.array([xy[0], xy[1], z]))
    qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
    data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
    state.info["terrain_type"] = jp.array(0, dtype=jp.int32)
    state.info["terrain_level_spawn"] = jp.array(0, dtype=jp.int32)
    state.info["rng"] = rng
    if "terrain_type" in state.metrics:
      state.metrics["terrain_type"] = jp.zeros(())
      state.metrics["terrain_level"] = jp.zeros(())
    obs = self._get_obs(data, state.info)
    return state.replace(data=data, obs=obs)  # pyrefly: ignore[missing-attribute]

  def _wrap_in_tile(self, state: mjx_env.State) -> mjx_env.State:
    """The floor is one map. Walking off a feature stays on that floor."""
    return state

  def _update_terrain_curriculum(
      self, info: dict[str, Any], tracking_lin: jax.Array, fell: jax.Array
  ) -> None:
    info["track_sum"] = info["track_sum"] + tracking_lin
    info["track_count"] = info["track_count"] + 1.0
    steps = info["steps"] if "steps" in info else jp.zeros(())
    timeout = steps >= (self._config.episode_length - 1)
    ending = fell | timeout
    mean = info["track_sum"] / jp.maximum(info["track_count"], 1.0)
    variable = info["terrain_type"] >= 2
    success = ending & variable & (~fell) & (mean >= _TRACK_SUCCESS)
    fail = ending & variable & (fell | (mean < _TRACK_FAIL))
    delta = success.astype(jp.int32) - fail.astype(jp.int32)
    level = info["AutoResetWrapper_preserve_info"]["terrain_level"]
    info["AutoResetWrapper_preserve_info"]["terrain_level"] = jp.where(
        ending, jp.clip(level + delta, 0, _N_LEVELS - 1), level
    )

  def reset(
      self, rng: jax.Array, terrain_type: Optional[str] = None
  ) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )

    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }
    if self._terrain_mix:
      info["terrain_type"] = jp.array(0, dtype=jp.int32)
      info["terrain_level_spawn"] = jp.array(0, dtype=jp.int32)
      info["track_sum"] = jp.array(0.0)
      info["track_count"] = jp.array(0.0)
      # Survives Brax's full reset. The wrapper copies this dict across episodes.
      info["AutoResetWrapper_preserve_info"] = {
          "terrain_level": jp.array(0, dtype=jp.int32),
      }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    if self._terrain_mix:
      metrics["terrain_type"] = jp.zeros(())
      metrics["terrain_level"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    if self._terrain_mix:
      rng, key = jax.random.split(state.info["rng"])
      state.info["rng"] = rng
      if terrain_type is None:
        state = self.place_terrain_mix(state, jp.array(0, dtype=jp.int32), key)
      else:
        if terrain_type not in _PINNED_TERRAIN_TYPE:
          raise ValueError(
              "terrain_type must be flat, bumps, slope, or stairs, "
              f"got {terrain_type!r}."
          )
        state = self.place_terrain_mix(
            state,
            jp.array(0, dtype=jp.int32),
            key,
            pinned=terrain_type,
        )
    return state

  # def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
  #   qpos = state.data.qpos
  #   new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
  #   new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
  #   qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
  #   state = state.replace(data=state.data.replace(qpos=qpos))
  #   return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._terrain_mix:
      state = self._wrap_in_tile(state)
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    if self._terrain_mix:
      self._update_terrain_curriculum(state.info, rewards["tracking_lin_vel"], done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    if self._terrain_mix:
      state.metrics["terrain_type"] = state.info["terrain_type"].astype(
          reward.dtype
      )
      state.metrics["terrain_level"] = state.info["terrain_level_spawn"].astype(
          reward.dtype
      )

    done = done.astype(reward.dtype)
    state = state.replace(  # pyrefly: ignore[missing-attribute]
        data=data, obs=obs, reward=reward, done=done
    )
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["command"],  # 3
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 12
        joint_vel,  # 12
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        # Ayush: gyro z is the yaw rate the tracking term already uses.
        "yaw_rate": self._cost_yaw_rate(
            info["command"], self.get_gyro(data)
        ),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_yaw_rate(
      self, commands: jax.Array, gyro: jax.Array
  ) -> jax.Array:
    # Ayush: penalize yaw rate only when the yaw command is about 0.
    # 0.01 matches the zero-command tolerance used by stand_still.
    # The scale in reward_config (-50) is the weight on yaw_rate^2.
    yaw_cmd = commands[2]
    yaw_rate = gyro[2]
    straight = jp.abs(yaw_cmd) < 0.01
    return jp.square(yaw_rate) * straight

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)  # pyrefly: ignore[missing-attribute]
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)  # pyrefly: ignore[missing-attribute]

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1
