# Reward Shaping and Reward Hacking in Lunar Lander 🚀

**A deep reinforcement learning experiment using PPO, Gymnasium, and custom reward functions.**

This project examines a simple but important question: **Does maximizing an AI agent's reward necessarily mean it accomplishes the task we intended?** I trained several Proximal Policy Optimization (PPO) agents to control a simulated Lunar Lander, deliberately introduced a reward-hacking incentive, and then tested two alternative reward designs: a timestep penalty and a precision-landing bonus.

The results show a striking difference between *high reward under a flawed objective* and *good performance on the actual landing task*.

## Highlights

- **Reward hacking:** a hover-bonus agent averaged **+771.06 modified reward**, but **−48.74 original reward**, **0/20 successful rests**, and **17/20 timeouts**.
- **Correction:** removing the hover bonus and adding a **−0.05 per-step penalty** produced **15/20 successful rests**.
- **Precision landing:** a separate model trained with a **+50 terminal bonus** for a successful rest at `|x| <= 0.15` achieved **20/20 centred rests** in the tested set of 20 evaluation seeds, compared with **5/20** for the timestep-penalty model.

> **Scope of the result:** These are observations from one training run per reward design on 20 evaluation seeds (`100`–`119`). They are not a guarantee of success on all initial conditions. The centred criterion measures the lander's **centre x-coordinate**, not full physical clearance of both landing legs between the flags.

## Research questions

1. Can PPO learn to land a rocket with Gymnasium's built-in rewards?
2. Can an additional per-step hover bonus make the agent game the reward function instead of landing?
3. Does removing the bonus and charging for time restore task performance?
4. Can a one-time bonus for **successful centred touchdown** improve landing precision?

## Environment and approach

- **Environment:** `LunarLander-v3` (Gymnasium / Box2D)
- **Algorithm:** Proximal Policy Optimization (PPO), via Stable-Baselines3
- **Policy:** `MlpPolicy` (a neural-network policy over eight observations)
- **Action space:** four discrete commands (no thrust, left engine, main engine, right engine)
- **Observations:** x/y position, x/y velocity, angle, angular velocity, and left/right leg contact
- **Training:** 500,000 environment timesteps per compared model
- **Evaluation:** deterministic actions, 20 environment seeds (`100`–`119`), and **the unmodified original environment reward** for cross-model reward comparisons

The neural network and PPO optimizer are implemented by Stable-Baselines3. My contributions are the experiment setup, custom reward wrappers, training/evaluation scripts, comparison methodology, and analysis.

## Reward designs

Let `r_env` be Gymnasium's original step reward, `t` the timestep, and `x` the normalized horizontal position.

| Experiment | Additional training reward | What it tests |
|---|---|---|
| **Original PPO** | None | Baseline performance |
| **Centering PPO** | Potential-based horizontal shaping | Whether encouraging horizontal progress improves alignment |
| **Hover PPO** | `+1` on each qualifying upright, slow, airborne step | Whether the policy exploits repeated payments instead of landing |
| **Corrected PPO** | `−0.05` each step | Whether removing the flawed incentive and penalizing delay restores landing behavior |
| **Precision PPO** | `−0.05` each step **and** `+50` on successful terminal rest at `|x| <= 0.15` | Whether explicitly rewarding accurate *completed* landings improves precision |

### 1. Centering reward

The first experiment used a potential function `Phi(s) = -5 * abs(x)` and a shaping term:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

The intention was to reward progress toward the centre rather than repeatedly paying the agent just for occupying a favourable position. The implementation sets terminal potential to zero. This experiment is not the final precision model.

### 2. Flawed hover reward (reward hacking)

```python
if upright and slow and airborne and not terminated:
    reward += 1.0
```

This rewards a *condition* that can persist for hundreds of frames, even without landing. A model can obtain a large modified score while repeatedly avoiding the terminal state. In this experiment, the agent averaged **819.8 qualifying bonus steps** and timed out in **17 of 20 episodes**.

### 3. Corrected reward

```python
modified_reward = original_reward - 0.05
```

The repeated hover bonus was removed and a small cost was added for each simulation step. This discourages unnecessarily prolonged episodes. It does **not**, by itself, ensure safe landings: an agent might also terminate quickly through a crash. That is why success and crash rates must be evaluated separately.

### 4. Precision terminal reward

```python
modified_reward = original_reward - 0.05
if successful_rest and abs(float(observation[0])) <= 0.15:
    modified_reward += 50.0
```

The bonus is awarded **once**, on the terminal step, only if the environment reports a successful rest (`+100` original terminal reward) and the final centre x-coordinate is within the chosen threshold. The terminal bonus does not accumulate while hovering. PPO must learn a trajectory of actions that leads to this outcome.

## Results

### A. Reward hacking and correction

All **mean original rewards** below were evaluated under Gymnasium's *unmodified* reward function. Training objectives differ; therefore, training rewards should **not** be compared directly.

| Metric (20 episodes) | Original PPO | Hover PPO | Corrected PPO |
|---|---:|---:|---:|
| Mean **original** reward | 125.53 | −48.74 | **163.92** |
| Mean episode length | 336.9 | 890.2 | 413.8 |
| Successful terminal rests | 12/20 | 0/20 | **15/20** |
| Crashes | 7/20 | 3/20 | **4/20** |
| Timeouts | 1/20 | 17/20 | **1/20** |
| Episodes scoring ≥200 | 11/20 | 0/20 | 10/20 |
| Mean `abs(final_x)` on successful rests | 0.2790 | N/A | **0.2341** |

Under the **modified hover reward**, the hover-trained model additionally recorded:

| Metric | Value |
|---|---:|
| Mean modified episode reward | **771.06** |
| Mean original episode reward | **−48.74** |
| Mean qualifying bonus steps | **819.8** |
| Timeouts | **17/20** |

The arithmetic explains the apparently inflated score: `−48.74 + 819.8 = 771.06`.

### B. Precision landing

Both models were evaluated with the *original environment*, using the same 20 environment seeds. A **centred landing** means a successful terminal rest with `abs(final_x) <= 0.15`.

| Metric (20 episodes) | Corrected PPO | Precision PPO |
|---|---:|---:|
| Successful rests | 15/20 | **20/20** |
| Centred successful rests | 5/20 | **20/20** |
| Crashes | 4/20 | **0/20** |
| Timeouts | 1/20 | **0/20** |
| Centred success rate (all episodes) | 25% | **100%** |
| Precision among successful rests | 33.3% | **100%** |

The precision model's final x-coordinates on these seeds ranged from **−0.099 to +0.144**, all satisfying the chosen `±0.15` threshold. Its reported episode rewards ranged from **238.8 to 312.4** under the original reward.

### What these results establish—and what they do not

- **Supported:** Under the tested conditions, the hover-bonus agent strongly preferred prolonged airborne behaviour, while the precision-bonus agent completed 20/20 centred terminal rests.
- **Not established:** The precision model has a universal 100% success rate, the bonus is independently responsible for every improvement, or both landing legs always physically fit between the flags.
- **Experimental limitations:** One trained policy per condition, a relatively small evaluation sample, and some models trained with potentially different initialization histories. The correction changed **two things at once**—removing the hover reward and adding the step penalty—so it does not isolate the independent causal effect of the penalty.

## Repository layout

```text
lunar-lander-reward-shaping/
├── README.md
├── requirements.txt
├── .gitignore
├── custom_reward.py        # Reward wrappers
├── train.py                # Choose baseline / centering / hover / corrected / precision
├── watch.py                # Watch saved policies interact with the environment
├── evaluate_all.py         # Original vs hover vs corrected comparison
├── evaluate_hover.py       # Original reward vs modified hover reward
├── evaluate_precision.py   # Centred-landing measurement
├── record.py               # Optional MP4 recordings
├── results/
│   ├── summary.csv
│   └── precision_seed_results.csv
├── models/                 # Put locally trained .zip checkpoints here (Git-ignored)
└── videos/                 # Saved MP4 demonstrations (Git-ignored)
```

The Python scripts here reconstruct the experiment workflow from the project notes. **Before publishing, reconcile them with your locally tested source files** and make sure the repository version reproduces your actual runs. Trained checkpoints and videos were not provided for this repository bundle.

## Setup (Windows PowerShell)

Python 3.13.3 was used in the project session. The project also ran on a Windows laptop with an Intel Core i5-1335U, 16 GB RAM, and integrated Intel graphics, using **CPU training**.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activating `.venv`, use Command Prompt's `.venv\Scripts\activate.bat` or adjust execution policy for the current process. For GPU-less training, `device="cpu"` is intentional.

### Train

Choose a model by setting `EXPERIMENT` in `train.py` to one of:

```text
baseline | center | hover | corrected | precision
```

Then run:

```powershell
python train.py
```

By default this creates a fresh PPO model and trains for `500_000` environment timesteps. `models/` is created automatically. For a quick installation smoke test, temporarily reduce `TOTAL_TIMESTEPS`, but don't compare such a model against the recorded 500k results.

> Training results are stochastic. Changing library versions, seeds, or hyperparameters can change the outcomes. Saving a training seed does not guarantee exact replication across all environments.

### Evaluate

```powershell
python evaluate_all.py
python evaluate_hover.py
python evaluate_precision.py
```

`evaluate_precision.py` accepts a `MODEL_PATH` constant, which you can switch between the corrected and precision checkpoints. All comparisons should use the same original environment and evaluation seeds.

### Watch / record

```powershell
python watch.py
python record.py
```

Select the checkpoint at the top of each script. The `watch.py` script uses `render_mode="human"` and the trained model's deterministic action prediction. `record.py` uses `render_mode="rgb_array"` with Gymnasium's `RecordVideo` wrapper.

## Future work

- Evaluate on **100+ additional unseen environment seeds**.
- Repeat each training configuration with **multiple independent training seeds**, then report variability and uncertainty intervals.
- Add an **ablation study**: original reward vs `−0.05` only vs `+50` precision only vs both.
- Measure final touchdown vertical speed, horizontal speed, angular stability, engine usage, and exact physical contact within the flags.
- Compare different precision thresholds and terminal bonus magnitudes, monitoring for new reward-hacking behaviours.

## References

- [Gymnasium — Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Gymnasium — Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
- [Gymnasium — Custom Wrappers](https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/)
- [Stable-Baselines3 — PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 — Installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- [Schulman et al., 2017 — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

**Takeaway:** Reward functions define what an RL agent is incentivized to do—not necessarily what its designer intends. Evaluating the original task, rather than the modified score alone, is essential for detecting reward hacking.
