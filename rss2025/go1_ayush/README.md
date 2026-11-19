# Go1 live viewer

Watch the stock flat-terrain policy in a MuJoCo window. With no controller, the command stays at `--vx --vy --yaw`. With an Xbox controller on `/dev/input/js0`, the sticks replace those flags. Close the window to stop. This does not save a video.

The window opens with both side panels hidden, so only the 3D view shows. Tab shows or hides the left panel. Shift-Tab shows or hides the right panel.

`which python` should print `/home/kingayush/King_stuff/RL_sim/mujoco_playground/rss2025/.venv/bin/python`.

```bash
cd /home/kingayush/King_stuff/RL_sim/mujoco_playground/rss2025
source .venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PATH="$PATH:/home/kingayush/miniforge3/bin"
which python
python go1_ayush/play_viewer.py \
  --checkpoint_path=/home/kingayush/King_stuff/RL_sim/logs/go1-stock/Go1JoystickFlatTerrain-20261111-221943/checkpoints/000206438400 \
  --vx=0.5 \
  --vy=0 \
  --yaw=0
```

`--vx`, `--vy`, and `--yaw` are optional. Those values are the defaults. The first launch compiles the policy and one env step before the window opens.

Pick a saved run with `--run`. The script prints the checkpoint path it resolved. If you also pass `--checkpoint_path`, that path wins.

```bash
python go1_ayush/play_viewer.py --run stock
python go1_ayush/play_viewer.py --run yaw
python go1_ayush/play_viewer.py --run latest
```

`--run stock` is the finished stock checkpoint. `--run yaw` is the finished yaw-penalty checkpoint. `--run latest` is the newest checkpoint under `~/King_stuff/RL_sim/logs`. With neither flag, the viewer loads stock.

`python go1_ayush/play_and_plot.py --run latest` is the same live window, and closing it writes `/home/kingayush/King_stuff/RL_sim/Sim_Videos/go1-push-controller.png`.

The window prints `Accepting commands.` when the control loop starts taking vx/vy/yaw. About once a second the terminal prints the command, the actual body vx/vy/yaw, the error (`cmd - act`), and the kick (`kick 0.0 N` when no shove is active). A kick lasts a fraction of a second, so while it is on the terminal also prints that force and direction every control step. An orange arrow on the trunk points the same way. Closing the window prints the mean over the run.

## Xbox controller

The device is `/dev/input/js0`. Check it before launching the viewer:

```bash
jstest --normal /dev/input/js0
```

`jstest` prints one line of axes and buttons. Ctrl-C stops it. On this Xbox pad the axes are left stick X, left stick Y, right stick X, right stick Y, then the two triggers. A resting stick is near 0. A resting trigger sits at -32767.

Stick map. Values are scaled to the ranges `joystick.py` samples (`command_config.a`: vx ±1.5 m/s, vy ±0.8 m/s, yaw ±1.2 rad/s). A stick inside the 0.15 deadzone sends 0.

- Left stick up: +vx (forward). Down: -vx.
- Left stick left: +vy. Right: -vy.
- Right stick left: +yaw. Right: -yaw.

If `/dev/input/js0` is missing, the script prints that and keeps `--vx --vy --yaw`. If the controller is connected, it prints that the sticks override the flags and reads them every frame. This pad also reports -1 on every stick axis when it is asleep. That is not a real push, so the flags stay until a stick moves off -1.
