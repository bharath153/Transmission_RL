# Technical Notes — Transmission Shift Control with Tabular Q-Learning

## Purpose

This repository demonstrates the mechanics of reinforcement learning for discrete transmission shift control in a simplified environment.

## Environment

The state contains:
- vehicle speed;
- gear;
- engine RPM;
- simplified engine torque.

Actions are hold, upshift and downshift.

The reward uses a simplified BSFC proxy together with a shift penalty.

## Learning Algorithm

The implementation is **tabular Q-learning** with:
- discretized state bins;
- epsilon-greedy exploration;
- learning rate 0.1;
- discount factor 0.95;
- 2,000 training episodes.

It is not PPO.

## Baseline

The learned policy is compared against a fixed RPM-threshold strategy.

## Limitations

The engine map, longitudinal dynamics, drive cycle and fuel-consumption representation are simplified. Reported savings apply only to this demonstration environment and should not be interpreted as validated vehicle fuel-economy gains.
