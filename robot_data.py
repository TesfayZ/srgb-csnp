"""
robot_data.py – Generate trajectories from MuJoCo robot models,
contact dynamics, and variable sampling.
Used by benchmark_robotics.py and ablation experiments.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def generate_robot_trajectories(model_xml, policy, horizon, n_trajectories,
                                noise_sigma=0.0, dt=0.01):
    """
    Generate trajectories from a MuJoCo robot model with a given policy.

    Raises RuntimeError if MuJoCo is unavailable or the model cannot be loaded.
    There is deliberately no synthetic fallback: a fabricated trajectory carries
    no physical invariant, so the pipeline recovers nothing and the run writes a
    real-looking 0% into Results/benchmark_robotics.csv that is indistinguishable
    from a genuine method failure. Failing loudly instead lets the orchestrator
    mark the benchmark as failed/skipped rather than silently reporting a fake
    number. (Note: on some CPUs `import mujoco` aborts with SIGILL rather than a
    catchable ImportError; that surfaces as a hard subprocess failure, which is
    equally honest.)
    """
    try:
        import mujoco
    except ImportError as e:
        raise RuntimeError(
            "MuJoCo is not installed, so no real robot trajectories can be "
            "generated. Install mujoco to run benchmark_robotics.py; the "
            "benchmark does not fabricate dummy data."
        ) from e

    try:
        model = mujoco.MjModel.from_xml_path(model_xml)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MuJoCo model '{model_xml}': {e}"
        ) from e

    model.opt.timestep = dt
    data = mujoco.MjData(model)
    trajectories = []
    for traj_idx in range(n_trajectories):
        mujoco.mj_resetData(model, data)
        # mj_resetData leaves qpos/qvel at the model's default (zero here,
        # no keyframe defined), so without an initial kick every trajectory
        # sits at the q=0, qdot=0 equilibrium forever under gravity=0 --
        # producing degenerate all-zero data with no dynamics to recover an
        # invariant from. Randomize the initial state per trajectory so the
        # system actually oscillates.
        data.qpos[:] = np.random.uniform(-1.0, 1.0, model.nq)
        data.qvel[:] = np.random.uniform(-1.0, 1.0, model.nv)
        traj = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}
        for step in range(horizon):
            state = np.concatenate([data.qpos, data.qvel])
            if noise_sigma > 0:
                state += np.random.normal(0, noise_sigma, len(state))
            action = policy(state)
            if model.nu > 0:
                data.ctrl[:] = action[:model.nu]
            mujoco.mj_step(model, data)
            next_state = np.concatenate([data.qpos, data.qvel])
            if noise_sigma > 0:
                next_state += np.random.normal(0, noise_sigma, len(next_state))
            reward = np.random.randn()  # dummy
            traj['states'].append(state)
            traj['actions'].append(action)
            traj['rewards'].append(reward)
            traj['next_states'].append(next_state)
        trajectories.append(traj)
    return trajectories

def detect_contacts(states, model_xml):
    """Placeholder for contact detection; returns boolean list."""
    return [False] * len(states)

def sample_with_variable_dt(trajectory, dt_distribution='exponential', scale=0.1,
                            min_dt=0.01, max_dt=0.3):
    """Downsample a trajectory with non-uniform time steps."""
    N = len(trajectory['states'])
    if dt_distribution == 'exponential':
        dt_samples = np.random.exponential(scale=scale, size=N-1)
    elif dt_distribution == 'uniform':
        dt_samples = np.random.uniform(min_dt, max_dt, N-1)
    else:
        raise ValueError(f"Unknown dt distribution: {dt_distribution}")
    times = np.concatenate([[0], np.cumsum(dt_samples)])
    resampled = {k: v.copy() for k, v in trajectory.items()}
    resampled['times'] = times
    return resampled