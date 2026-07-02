"""
Transmission Shift Point Optimizer — Reinforcement Learning (PPO-style)
========================================================================
Trains an RL agent to determine optimal gear shift points for minimum
fuel consumption, modelled as a 1D discrete action environment.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import os, json

np.random.seed(7)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Vehicle & Engine Parameters ──
GEARS  = 6
RATIOS = np.array([4.17, 2.34, 1.52, 1.14, 0.87, 0.69])
FINAL_DRIVE = 3.73
WHEEL_RADIUS = 0.316   # m
ENGINE_REDLINE = 6500  # RPM
IDLE_RPM       = 800

def engine_torque(rpm):
    """Simplified torque curve (peak at ~3500 rpm)."""
    peak_rpm = 3500
    return 180 * np.exp(-((rpm - peak_rpm)**2) / (2*1500**2)) + 60

def fuel_consumption(rpm, torque, gear):
    """BSFC-proxy (g/kWh) — lowest around 2000-3000 rpm mid-load."""
    bsfc_base = 260 + 0.015*(rpm-2500)**2/1000 + max(0, 0.2*(gear-4))**2
    return bsfc_base

class TransmissionEnv:
    def __init__(self):
        self.gear   = 1
        self.speed  = 20.0  # km/h
        self.steps  = 0
        self.max_steps = 200
    def reset(self):
        self.gear  = 1; self.speed = 20.0; self.steps = 0
        return self._obs()
    def _obs(self):
        rpm = self._rpm()
        torque = engine_torque(rpm)
        return np.array([self.speed, self.gear, rpm, torque])
    def _rpm(self):
        wheel_rps = (self.speed / 3.6) / (2*np.pi*WHEEL_RADIUS)
        return np.clip(wheel_rps * RATIOS[self.gear-1] * FINAL_DRIVE * 60, IDLE_RPM, ENGINE_REDLINE)
    def step(self, action):
        # action: 0=hold, 1=shift_up, 2=shift_down
        if action == 1 and self.gear < GEARS:
            self.gear += 1
        elif action == 2 and self.gear > 1:
            self.gear -= 1
        self.speed = min(self.speed + np.random.uniform(-1, 2), 130)
        self.speed = max(self.speed, 10)
        rpm = self._rpm()
        torque = engine_torque(rpm)
        fc = fuel_consumption(rpm, torque, self.gear)
        reward = -fc / 1000.0 + (0.1 if action == 0 else -0.05)
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._obs(), reward, done

# ── Simulate training: RL agent (tabular Q-learning proxy) ──
env = TransmissionEnv()
n_episodes = 2000
lr = 0.1; gamma_r = 0.95; eps = 1.0; eps_min = 0.05; eps_decay = 0.995

# Discretise obs for tabular Q
def discretize(obs):
    spd = int(obs[0] / 10)
    gear = int(obs[1]) - 1
    rpm_bin = int(np.clip(obs[2]/1000, 0, 6))
    return (min(spd,12), gear, rpm_bin)

Q = {}
def get_q(s, a): return Q.get((s,a), 0.0)

episode_rewards = []
shift_efficiencies = []

for ep in range(n_episodes):
    obs = env.reset(); total_r = 0; shifts = 0
    s = discretize(obs)
    for _ in range(200):
        if np.random.rand() < eps:
            action = np.random.randint(0, 3)
        else:
            action = max(range(3), key=lambda a: get_q(s, a))
        obs2, r, done = env.step(action)
        s2 = discretize(obs2)
        best_next = max(get_q(s2, a) for a in range(3))
        Q[(s, action)] = get_q(s, action) + lr * (r + gamma_r*best_next - get_q(s, action))
        s = s2; total_r += r
        if action in [1,2]: shifts += 1
        if done: break
    eps = max(eps_min, eps * eps_decay)
    episode_rewards.append(total_r)
    shift_efficiencies.append(shifts)

print(f"Final avg reward (last 200 ep): {np.mean(episode_rewards[-200:]):.4f}")

# ── Simulate baseline (fixed threshold) vs RL ──
def baseline_policy(obs):
    rpm = obs[2]
    if rpm > 3500: return 1
    if rpm < 1500: return 2
    return 0

def run_episode(policy_fn):
    env2 = TransmissionEnv(); obs = env2.reset(); total = 0
    gear_trace, rpm_trace, speed_trace, fc_trace = [], [], [], []
    for _ in range(200):
        action = policy_fn(obs)
        obs, r, done = env2.step(action)
        gear_trace.append(obs[1]); rpm_trace.append(obs[2])
        speed_trace.append(obs[0])
        fc = fuel_consumption(obs[2], engine_torque(obs[2]), obs[1])
        fc_trace.append(fc); total += r
        if done: break
    return total, np.array(gear_trace), np.array(rpm_trace), np.array(speed_trace), np.array(fc_trace)

def rl_policy(obs):
    s = discretize(obs)
    return max(range(3), key=lambda a: get_q(s, a))

_, g_rl, r_rl, s_rl, f_rl  = run_episode(rl_policy)
_, g_bl, r_bl, s_bl, f_bl  = run_episode(baseline_policy)
ts = np.arange(len(g_rl))

json.dump({"mean_reward_last200": round(float(np.mean(episode_rewards[-200:])),4),
           "fuel_saving_pct": round(float((f_bl.mean()-f_rl.mean())/f_bl.mean()*100),2)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Training reward curve
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(episode_rewards, alpha=0.25, color='#90CAF9', lw=0.8)
ax.plot(np.arange(window-1, n_episodes), smoothed, color='#1565C0', lw=2, label=f'Moving avg ({window} ep)')
ax.set_xlabel('Episode', fontsize=11); ax.set_ylabel('Total Reward', fontsize=11)
ax.set_title('Q-Learning Training Curve — Transmission Optimizer', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_training_curve.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Gear vs RPM traces
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle('RL Agent vs Baseline — Drive Cycle Comparison', fontsize=14, fontweight='bold')
axes[0].plot(ts, s_rl, color='#1976D2', label='RL Agent'); axes[0].plot(ts, s_bl, color='#FF5722', ls='--', label='Baseline')
axes[0].set_ylabel('Vehicle Speed (km/h)', fontsize=10); axes[0].legend()
axes[1].step(ts, g_rl, color='#1976D2', label='RL Gear'); axes[1].step(ts, g_bl, color='#FF5722', ls='--', label='Baseline Gear')
axes[1].set_ylabel('Gear', fontsize=10); axes[1].set_yticks(range(1, 7)); axes[1].legend()
axes[2].plot(ts, f_rl, color='#1976D2', label=f'RL BSFC (mean={f_rl.mean():.0f})')
axes[2].plot(ts, f_bl, color='#FF5722', ls='--', label=f'Baseline BSFC (mean={f_bl.mean():.0f})')
axes[2].set_ylabel('BSFC (g/kWh)', fontsize=10); axes[2].set_xlabel('Time Step', fontsize=10); axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_drive_cycle_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Fuel saving bar chart
labels = ['Baseline\n(Fixed Threshold)', 'RL Agent\n(Optimal)']
vals   = [f_bl.mean(), f_rl.mean()]
saving = (f_bl.mean()-f_rl.mean())/f_bl.mean()*100
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, vals, color=['#FF8A65','#1976D2'], width=0.4, edgecolor='white')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean BSFC (g/kWh)', fontsize=11)
ax.set_title(f'Fuel Consumption Reduction\n▼ {saving:.1f}% via RL Shift Optimization', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_fuel_saving.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Q-value heatmap for gear 3
fig, ax = plt.subplots(figsize=(8, 5))
speeds = range(0, 13); rpms = range(0, 7)
qmap = np.zeros((7, 13))
for si, sp in enumerate(speeds):
    for ri, rp in enumerate(rpms):
        state = (sp, 2, rp)  # gear index 2 = gear 3
        best_a = max(range(3), key=lambda a: get_q(state, a))
        qmap[ri, si] = best_a
sns.heatmap(qmap, ax=ax, cmap='RdYlGn', vmin=0, vmax=2,
            xticklabels=[f'{s*10}' for s in speeds], yticklabels=[f'{r}k' for r in rpms],
            cbar_kws={'ticks':[0,1,2], 'label':'Action (0=Hold, 1=Up, 2=Down)'})
ax.set_xlabel('Speed (km/h)', fontsize=10); ax.set_ylabel('RPM bin (×1000)', fontsize=10)
ax.set_title('Learned Q-Policy — Gear 3 State Space', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_q_policy.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")
