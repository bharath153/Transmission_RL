# Interview Q&A — Transmission Shift Point Optimizer — Reinforcement Learning
## Tabular Q-Learning (proxy for PPO) | Powertrain Control & Transmission

### Q1: Why RL for transmission control vs classical optimal control?

Classical methods (Dynamic Programming, Model Predictive Control) require an accurate system model and are computationally expensive for online use. RL learns directly from interaction without an explicit model, making it more robust to model uncertainties and adaptable to different drivers or road conditions. Volkswagen and ZF have published research on RL-based AMT control (ZF Friedrichshafen AG, 2021 SAE paper).

---

### Q2: What is BSFC and why minimize it?

Brake-Specific Fuel Consumption (g/kWh) is the mass of fuel consumed per unit of mechanical power output. It maps the engine's efficiency across its operating map (speed × torque). Shifting to keep the engine in its minimum-BSFC island (typically 2000–3000 RPM at 50-80% load for most petrol engines) is the core goal of any fuel-optimal shift strategy.

---

### Q3: What is the difference between Q-Learning and PPO?

Q-Learning is a model-free, value-based, tabular RL algorithm. It works well in discrete, low-dimensional state spaces but scales poorly to continuous or high-dimensional spaces. PPO (Proximal Policy Optimization) is a policy-gradient method that uses neural networks to represent the policy and handles continuous state/action spaces natively. For a real transmission controller with continuous signals from dozens of sensors, PPO (or SAC) would be the production choice.

---

### Q4: How is the reward function designed?

The reward is a linear combination of negative BSFC (lower fuel consumption = higher reward) and a smoothness penalty for excessive gear changes. The balance between these terms is a hyperparameter (reward shaping). Poorly designed rewards can lead to gear hunting — rapid up/down shifting — so the penalty term is critical for smooth, realistic behaviour.

---

### Q5: What data would you need for a real deployment?

For a real vehicle: (1) Engine torque and BSFC maps from dyno testing, (2) CAN bus signals (engine speed, torque demand, vehicle speed, accelerator pedal position, gear), (3) Real driving cycle data (WLTP, RDE) for training and validation, (4) Hardware-in-the-Loop (HiL) test rig validation before on-vehicle trials.

---

