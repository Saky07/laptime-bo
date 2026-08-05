# Lap-Time Tuning via Multi-Fidelity Bayesian Optimization

Data-efficient tuning of an autonomous (Formula Student Driverless) race car's
parameters to minimize lap time, using Bayesian optimization. A cheap, approximate
simulator is combined with expensive high-fidelity evaluations so the optimizer
reaches a near-optimal setup using as few expensive runs as possible.

> Status: work in progress.

## The idea
Lap time is a black-box function of tunable settings (racing-line aggression,
braking, cornering-speed limits, ...). Each evaluation is expensive (track time,
crash risk). Bayesian optimization finds good settings in very few evaluations;
the multi-fidelity variant spends expensive evaluations only where they matter.

## Repo layout
```
laptime-bo/
├── src/             # source code (simulator, BO, experiments)
├── results/         # generated plots
├── requirements.txt # dependencies
└── README.md
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/simulator.py         # sanity-check the lap-time simulator
```

## Results
<!-- The headline plot goes here once experiments are run. -->

## License
MIT
