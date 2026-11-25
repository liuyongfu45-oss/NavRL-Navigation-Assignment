# NavRL — Multi-Robot Navigation in Static, Dynamic and Mixed Obstacle Scenarios

A Navigation Reinforcement Learning (NavRL) assignment: differential-drive robots
learn to reach assigned goal points while avoiding static obstacles, moving
obstacles and other robots. This repository contains the simulation package, the
reward design, the actor–critic neural network, the evaluation metrics, the test
suite and the generated result tables and figures.

## Demonstration videos

**S1 — Dynamic obstacles**

https://github.com/user-attachments/assets/ce9fa014-11a0-495e-a97c-c7d62fb08013

**S2 — Static obstacles**

https://github.com/user-attachments/assets/d104350e-fd30-48d0-a2ca-e94eb2f5b762

**S3 — Mixed static + dynamic**

https://github.com/user-attachments/assets/08e59ef2-8121-4927-a02f-97e70607d9ce

| Item | Value |
|---|---|
| Course | Deep Reinforcement Learning for Autonomous Robot Navigation (NavRL) |
| Student | LIU YONGFU |
| Student ID | 20263321 |
| Date | 2026-11-18 |
| Language | Python 3 (NumPy / PyTorch CPU / Matplotlib) |
| Recorded demos | `NavRL_S1_Dynamic.mp4`, `NavRL_S2_Static.mp4`, `NavRL_S3_Mixed.mp4` (720×720, 30 fps, 8.00 s each) |

## Assignment coverage

The assignment asks for four components; each is mapped to code and documentation:

| # | Requirement | Where it is covered |
|---|---|---|
| 1 | Three different navigation scenarios (multi-robot, dynamic, static, mixed) | `navrl/scenarios.py`, `configs/scenarios.json`, Section [2](#2-scenarios) |
| 2 | Reward function: how it works, positive/negative reward table | `navrl/reward.py`, Section [4](#4-reward-function) |
| 3 | Performance metrics: tables, graphs, numerical values | `navrl/metrics.py`, `results/`, Section [5](#5-evaluation-metrics-and-results) |
| 4 | Neural network hyperparameters and which networks are used | `navrl/network.py`, Section [6](#6-neural-network-and-hyperparameters) |

## 1. Repository layout

```
NavRL-Navigation-Assignment/
├── README.md                     # This document (explanation + results)
├── requirements.txt              # Dependencies (torch optional for training)
├── configs/
│   └── scenarios.json            # World, robot, reward and scenario configuration
├── navrl/                        # The importable Python package
│   ├── environment.py            # 2-D multi-robot navigation simulator (MDP)
│   ├── reward.py                  # Reward terms, RewardConfig, reward table
│   ├── scenarios.py               # Deterministic S1/S2/S3 layout generation
│   ├── policy.py                 # APF clearance-aware baseline + random policy
│   ├── network.py                 # CNN/MLP encoders, actor–critic, PPO agent
│   ├── metrics.py                 # Metrics, aggregation, CSV export, figures
│   ├── evaluate.py                # Roll out every scenario and summarise
│   ├── train.py                   # PPO training loop + convergence curve
│   └── utils.py                   # Config loading, seeding, math helpers
├── tests/                         # Unit test suite (unittest, no pytest required)
│   ├── test_scenarios.py
│   ├── test_environment.py
│   ├── test_reward.py
│   ├── test_policy.py
│   ├── test_network.py
│   └── test_metrics.py
├── scripts/
│   ├── run_evaluation.py          # CLI: run episodes, write CSV + figures
│   └── run_demo.py                # CLI: one short episode per scenario
└── results/
    ├── scenario_summary.csv       # Aggregate metrics per scenario
    ├── episode_detail.csv         # Per-robot per-episode measurements
    └── figures/                   # Generated comparison graphs
```

## 2. Scenarios

The same simulator, reward and policy interface are used in every scenario; only
the obstacle layout changes, which tests whether the behaviour generalises
beyond one favourable arrangement.

| Scenario key | Title | Robots | Static obstacles | Dynamic obstacles | Source video |
|---|---|---:|---:|---:|---|
| `S1_dynamic` | Dynamic obstacles | 3 | 0 | 14 | `NavRL_S1_Dynamic.mp4` |
| `S2_static` | Static obstacles | 3 | 12 | 0 | `NavRL_S2_Static.mp4` |
| `S3_mixed` | Mixed static + dynamic | 3 | 7 | 8 | `NavRL_S3_Mixed.mp4` |
| `assignment_multi_robot` | Assignment variant: 6 robots, 15 obstacles (mixed) | 6 | 8 | 7 | — |

The `assignment_multi_robot` variant implements the literal assignment statement
of **six robots and fifteen obstacles** in a mixed layout; the three recorded
videos use a three-robot camera view of the same world.

### 2.1 World and robot configuration

| Parameter | Value | Meaning |
|---|---:|---|
| World size | 36 × 36 m | Square planar world, coordinates in `[-18, 18]` |
| Robot radius | 0.35 m | Circular robot body |
| Max linear velocity | 2.0 m/s | Upper bound on commanded speed |
| Max angular velocity | 1.5 rad/s | Upper bound on commanded turn rate |
| Arrival radius | 0.40 m | Goal reached when distance is below this |
| Safety distance | 1.20 m | Desired clearance from obstacles |
| Control time step `DT` | 0.1 s | Integration interval between decisions |
| Max steps per episode | 300 | Timeout horizon (recorded video: 4000 frames) |

### 2.2 Observed outcomes in the submitted videos

Read from the final frame of each recording (step 234, 18.72 s simulation time):

| Video | Obstacles | Reached | R1 reward | R2 reward | R3 reward | Observation |
|---|---|---:|---:|---:|---:|---|
| `NavRL_S1_Dynamic.mp4` | 14 dynamic | 3/3 | +8.63 | +8.62 | +8.60 | All robots cross the moving field to their stars |
| `NavRL_S2_Static.mp4` | 12 static | 3/3 | +8.91 | +8.03 | +8.03 | Straightest routes; highest static-field rewards |
| `NavRL_S3_Mixed.mp4` | 7 static + 8 dynamic | 3/3 | +1.84 | +8.30 | +8.07 | R1 detours near a moving obstacle, hence its lower reward |

## 3. MDP formulation and observation

The navigation task is formulated as an MDP tuple `(S, A, P, r, γ)`. At each
step the environment returns a joint observation; the agent selects bounded
velocities; the environment transitions and returns a scalar reward. The
objective is to maximise the expected discounted return
`G_t = E[Σ γ^k r_{t+k}]`.

Each robot receives four observation groups, expressed in a goal-oriented frame:

| Observation | Shape | Contents |
|---|---|---|
| Internal state | `(8,)` | cos θ, sin θ, linear velocity, angular velocity, goal distance, goal bearing (cos, sin), robot radius |
| Goal direction | `(3,)` | Unit vector toward goal (x, y) and goal distance |
| Static rays | `(36, 4)` | 36 ray bearings × 4 channels: normalised distance, hit flag, relative bearing, clearance margin |
| Dynamic obstacles | `(5, 10)` | Up to 5 nearest moving bodies; each has relative position, relative velocity, distance, bearing, radius, relative speed, time-to-collision, mask |

Obstacles beyond the five-nearest limit are zero-filled so the tensor shape stays
fixed at deployment time. Other robots are treated as dynamic moving bodies.

## 4. Reward function

### 4.1 How it works

The reward combines two mechanisms:

1. **Terminal events** — a fixed one-off reward ends the episode: goal reached,
   collision, or timeout. Terminal events override the per-step shaping.
2. **Per-step shaping** — continuous dense feedback for goal progress, static
   clearance, dynamic clearance, motion smoothness and anti-stall.

```
r_t = r_progress + r_static + r_dynamic + r_smooth + r_stall
      + r_goal + r_collision + r_timeout
```

Terminal if/else rewards are added only when their event occurs; the shaping
terms are evaluated every step. The reward is balanced: the goal bonus is large
enough to be worth pursuing, while collision and timeout penalties are large
enough that reckless contact or idle waiting cannot become the highest-return
action. Motion is never rewarded for its own sake: a robot that moves without
approaching the goal earns no progress reward and still pays the safety and
smoothness costs.

### 4.2 Positive and negative reward table

| Reward term | Sign | Condition / formula | Magnitude | Behaviour trained |
|---|---|---|---:|---|
| Goal-reaching terminal reward | Positive | Distance to goal < arrival threshold (0.4 m) | `+10.0` | Arrive and stop at the goal; end episode successfully |
| Collision penalty (terminal) | Negative | Body overlaps an obstacle, another robot, or leaves the valid region | `-10.0` | Avoid contact with obstacles, robots and boundaries |
| Timeout penalty (terminal) | Negative | Step count reaches `max_steps` without reaching the goal | `-10.0` | Discourage waiting, stalling and indefinite wandering |
| Goal progress / velocity alignment | Positive when closer, negative when farther | `k_p · (d_prev − d_t)` | `k_p = 1.0` | Move toward the goal and align heading with the target |
| Static safety reward | Positive when far, negative when close | `+k_safe` if `d ≥ d_safe`; otherwise `-k_close · (d_safe − d)/d_safe` | `+0.15 / −0.60` per unit intrusion | Keep clearance from walls and stationary obstacles |
| Dynamic safety reward | Positive when far, negative when close | `+k_safe` if `d ≥ d_safe`; otherwise `-k_close · (d_safe − d)/d_safe` | `+0.15 / −0.60` per unit intrusion | Maintain clearance from moving robots or people |
| Smoothness reward | Negative | `-k_m · (|v_t − v_{t−1}| + |ω_t − ω_{t−1}|)` | `k_m = 0.05` | Discourage jerky, abrupt command changes; low-jerk motion |
| Anti-stall penalty | Negative | Far from goal but commanded speed near zero | `-0.10` per stalled step | Prevent the freezing failure observed beside a goal |

The same table is generated from `REWARD_TABLE` in `navrl/reward.py` and is
checked by `tests/test_reward.py`.

## 5. Evaluation metrics and results

### 5.1 Metric definitions

| Metric | Definition | Meaning |
|---|---|---|
| Success rate | Robots reaching the goal / total robot-episodes | Fraction of trials that complete the task |
| Collision rate | Robots with a contact / total robot-episodes | Frequency of safety failures |
| Completion time | Mean steps × `DT` until arrival | How quickly the task is finished |
| Path length | Cumulative per-step displacement | Efficiency and directness of the route |
| Total episode reward | Sum of `r_t` over the episode | Overall trajectory quality under the reward model |
| Minimum clearance | Closest approach to any obstacle over the episode | Margin of safety near obstacles |
| Control effort | `Σ ‖v‖² · DT` | Controller workload / energy use |
| Smoothness (jerk) | Sum of frame-to-frame command changes | Motion smoothness |

### 5.2 Aggregate results (30 episodes per scenario, APF baseline)

Results are written to `results/scenario_summary.csv`; per-robot measurements
are in `results/episode_detail.csv`.

| Scenario | Robot-episodes | Success % | Collision % | Time (s) | Path (m) | Reward | Clearance (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1_dynamic | 90 | 90.0 | 7.8 | 9.17 ± 9.20 | 14.07 ± 17.50 | 46.40 ± 41.49 | 3.26 |
| S2_static | 90 | 98.9 | 0.0 | 9.25 ± 9.20 | 14.62 ± 17.76 | 49.99 ± 42.83 | 3.65 |
| S3_mixed | 90 | 97.8 | 1.1 | 9.38 ± 9.27 | 14.68 ± 17.82 | 49.86 ± 43.19 | 3.15 |
| assignment_multi_robot | 180 | 81.1 | 15.6 | 19.65 ± 5.53 | 32.23 ± 8.44 | 83.11 ± 25.83 | 0.69 |

Interpretation: the static scenario is the safest and most direct; the dynamic
scenario has the highest contact rate because obstacle future positions cannot
be read from a single frame; the mixed scenario sits between them; the dense
six-robot variant has the tightest clearance and the most inter-robot crowding,
which is where the learned PPO policy is expected to improve over the baseline.

### 5.3 Figures

| Figure | File | What it shows |
|---|---|---|
| Success / collision rates | `results/figures/success_collision.png` | Bar chart of success and collision percentage per scenario |
| Reward and clearance | `results/figures/reward_clearance.png` | Mean episode reward against mean safety clearance |
| Time and path length | `results/figures/time_path.png` | Completion time and route length per scenario |
| Reward curves | `results/figures/reward_curves.png` | Cumulative reward over time for each scenario |
| Training convergence | `results/figures/training_curve.png` | PPO return and success rate over training iterations |

## 6. Neural network and hyperparameters

### 6.1 Networks identified

The learned controller is an **actor–critic network trained with Proximal Policy
Optimization (PPO)**. At deployment only the actor is used to infer motion
commands. Three sub-networks are composed:

* **2-D CNN encoder** — processes the static ray matrix `(36 × 4)` into a
  compact static-context embedding.
* **MLP encoder** — processes the nearest dynamic-obstacle matrix `(5 × 10)`
  into a dynamic-context embedding.
* **Shared MLP** — concatenates internal state, goal direction and the two
  embeddings; an actor head and a critic head branch from the shared features.

```
                 ┌──────────────────────┐
  internal (8) ─┤                      ├─▶ Shared MLP ─▶ Actor head ─▶ Beta(α, β) ─▶ [v, ω]
  goal dir (3) ─┤   Shared feature      │        128-128-64
  rays (36×4) ─▶▶ CNN encoder           ├─▶ Critic head ─▶ V(s)
  dynamic(5×10)▶▶ MLP encoder           │
                 └──────────────────────┘
```

The actor outputs the parameters of a **Beta distribution** per action dimension.
Because the action space is bounded on every axis, Beta is preferred to a
Gaussian: it stays inside `[0, 1]` and handles the endpoints without clipping.
During training actions are sampled from Beta for exploration; at deployment
the Beta mean is used. The critic ends in a single linear unit estimating `V(s)`.

### 6.2 Layer structure

| Part | Layer / units | Activation | Role |
|---|---|---|---|
| Static encoder | Conv2d(1→8, 3×3) → Conv2d(8→16, 3×3) → Linear 64 | ReLU | Extract static obstacle context |
| Dynamic encoder | Linear 50→64 → Linear 64→32 | ReLU | Encode nearest moving obstacles |
| Shared hidden 1 | Fully connected, 128 units | ReLU | Combine state with obstacle embeddings |
| Shared hidden 2 | Fully connected, 128 units | ReLU | Higher-level navigation features |
| Shared hidden 3 | Fully connected, 64 units | ReLU | Compress features toward the action |
| Actor head | 4 units (2 dims × α, β) | Beta distribution | Bounded velocity distribution |
| Critic head | 1 linear unit | None | State value estimate `V(s)` |

Total trainable parameters in this implementation: **193,029**.

### 6.3 Hyperparameters

| Parameter | Symbol | Value | Role |
|---|---|---:|---|
| Discount factor | `γ` | 0.99 | Weighs future rewards; closer to 1 favours long-term return |
| RL algorithm | — | PPO, actor–critic | Optimise the actor against advantage; critic estimates value |
| Advantage estimator | — | GAE (λ = 0.95) | Low-variance advantage between critic value and realised return |
| Optimizer | — | Adam | Gradient update for actor and critic |
| Learning rate | `lr` | `5 × 10⁻⁴` | Step size for policy and value updates |
| PPO clip ratio | — | 0.1 | Limits the per-update policy change |
| Critic loss | — | Huber loss | Robust value regression under large return outliers |
| Entropy coefficient | — | 0.01 | Encourages exploration during training |
| Action distribution | — | Beta(α, β) | Bounded, unbiased action distribution over `[0, 1]` |
| Max linear velocity | `v_lim` | 2.0 m/s | Upper bound on commanded speed after action scaling |
| Control time step | `DT` | 0.1 s | Simulation / integration interval between decisions |
| Max frames per episode | `FT` | 300 (quick demos); 4000 (recorded video) | Step limit before a timeout is judged |
| Action dimension | — | 2 | Linear + angular velocity |
| Hidden widths | — | 128 – 128 – 64 | Shared MLP feature dimensions |
| Deployment device | — | CPU | Inference runs on CPU in this environment |

## 7. Installation and usage

### 7.1 Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch is only required for PPO training and the network tests; the
environment, reward, metrics and APF baseline run with NumPy and Matplotlib.

### 7.2 Run the test suite

```bash
python -m unittest discover -s tests -v
```

The suite contains 34 tests covering scenario layouts, environment dynamics,
reward signs and components, baseline policies, network shapes and PPO update,
and metrics aggregation / CSV / figure export. Network tests are skipped
automatically if PyTorch is not installed.

### 7.3 Evaluate all scenarios

```bash
python scripts/run_evaluation.py --episodes 30
```

This writes `results/scenario_summary.csv`, `results/episode_detail.csv` and the
figures in `results/figures/`.

### 7.4 Run a short demonstration

```bash
python scripts/run_demo.py
```

### 7.5 Train the PPO agent

```python
from navrl.train import train, plot_training
from navrl.utils import load_config

config = load_config()
history = train(config, "S3_mixed", iterations=200, rollout_steps=120)
plot_training(history, "results/figures/training_curve.png")
```

## 8. Test inventory

| Test file | What is verified |
|---|---|
| `test_scenarios.py` | S1/S2/S3 obstacle counts, six-robot 15-obstacle variant, deterministic layouts, unknown-scenario error |
| `test_environment.py` | Observation tensor shapes, step advance, goal-reached terminal, collision detection, dynamic obstacle motion, action clipping |
| `test_reward.py` | Goal bonus positive, collision/timeout penalties negative, progress sign, safety signs, smoothness and stall penalties, component sum, reward table completeness |
| `test_policy.py` | APF and random actions are bounded and finite; APF moves toward the goal |
| `test_network.py` | Forward-pass shapes, Beta samples in `[0, 1]`, positive parameter count, PPO update produces finite losses (skipped without PyTorch) |
| `test_metrics.py` | Success/collision aggregation, CSV round-trip, figure generation |

## 9. Limitations

* **Simplified 2-D kinematics.** The simulator integrates point velocities with
  circular obstacles; it does not model full UAV dynamics, inertia or 3-D
  geometry, so results are indicative rather than a reproduction of the original
  Isaac Sim statistics.
* **Contact definition.** A contact is a geometric inflation test rather than a
  physical impact; one encounter spanning several frames is counted per frame.
* **Baseline evaluation.** The committed result tables use the APF
  clearance-aware baseline (the same policy tag shown in the videos); the PPO
  training loop is included and can be trained further to convergence.
* **Single recording per scenario.** A statistically reliable failure rate
  requires repeated episodes with varied seeds and layouts, which the
  evaluation script supports via `--episodes`.

## 10. References

1. Z. Xu, X. Han, H. Shen, H. Jin, K. Shimada. *NavRL: Learning Safe Flight in
   Dynamic Environments.* IEEE Robotics and Automation Letters, 2025, 10(4):
   3668–3675. DOI: 10.1109/LRA.2025.3546069.
2. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy
   Optimization Algorithms.* arXiv:1707.06347, 2017.
3. J. van den Berg, M. Lin, D. Manocha. *Reciprocal n-Body Collision Avoidance.*
   ISRR, 2011.
4. Official NavRL repository: https://github.com/Zhefan-Xu/NavRL
