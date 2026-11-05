# Transmission Shift-Point Optimizer — Tabular Q-Learning

> Learning discrete gear-shift actions in a simplified transmission environment

## Overview

This project demonstrates reinforcement learning for transmission shift control using a custom, lightweight vehicle/engine environment.

The agent observes:
- vehicle speed;
- current gear;
- engine speed;
- simplified engine torque.

Available actions:
- hold gear;
- shift up;
- shift down.

The objective is represented through a reward based on a simplified BSFC proxy plus a shift penalty.

## Reinforcement-Learning Method

The implementation uses **tabular Q-learning**, not PPO.

```text
Episodes: 2000
Learning rate: 0.10
Discount factor: 0.95
Exploration: epsilon-greedy
Actions: hold / upshift / downshift
```

Continuous observations are discretized into speed, gear and RPM bins.

## Baseline Comparison

The learned policy is compared with a fixed-threshold baseline:
- upshift above 3500 rpm;
- downshift below 1500 rpm;
- otherwise hold.

The comparison reports the mean simplified BSFC proxy and the resulting percentage reduction.

## Model Scope

This is a **demonstration environment**, not a high-fidelity powertrain model. The engine torque curve, BSFC relationship and drive evolution are simplified and synthetic.

The project demonstrates:
- Q-learning mechanics;
- state discretization;
- epsilon-greedy exploration;
- baseline policy comparison;
- policy visualization.

It does not claim production fuel-economy performance or transmission-controller validation.

## Outputs

The script exports:
- training reward history;
- learned-policy vs. baseline traces;
- simplified fuel-consumption comparison;
- Q-policy heatmap;
- metrics in `metrics.json`.

## Run

```bash
git clone https://github.com/bharath153/Transmission_RL.git
cd Transmission_RL
pip install -r requirements.txt
python src/train_rl.py
```

## Repository Structure

```text
Transmission_RL/
├── src/
│   └── train_rl.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Next Steps

For a higher-fidelity study, this framework could be extended with:
- a calibrated engine fuel-consumption map;
- prescribed drive cycles;
- longitudinal vehicle dynamics;
- shift-duration and drivability constraints;
- continuous-state RL methods;
- policy evaluation over multiple randomized episodes.
