# FISH: Formation of Intelligent Swarming Habits

We present a comprehensive approach to learn from the behavior of living fish and engineer the behavior of robot collectives. To this end, we built a three dimensional simulator that realistically captures the physics of BlueBots, an underwater robot collective designed to study swarm intelligence. With that simulator, we extended existing methods of learning from observation to create replica fish that could mimic aggregation and dispersion behavior without being explicitly programmed to do so. Finally, we investigated algorithms that lead robotic fishes towards efficient swimming formations, based on the assumption that a reduction of total energy expenditure is possible when swimming in a school. Our results include algorithms that successfully lead to schooling and orbiting behaviors, and learn aggregation and dispersion from observation. In the future, BlueBots will be used to test our algorithms and transfer them into the real world, where swarms of robotic fishes could search for crashed aircraft or sample environmental data more efficiently.

## Requirements

- Python 3.6
- Jupyter 1.0
- Numpy
- Scipy
- Matplotlib
- (PIP _not mandatory but recommended_)

## Installation

Either install Jupyter, Numpy, Scipy, and Matplotlib via PIP:

```
git clone https://code.harvard.edu/flb979/FISH && cd FISH
pip install -r ./requirements.txt
```

Or manually via https://jupyter.org/install and https://scipy.org/install.html

## Run

Move to one of the project folders (BlueSim, TuringFish, FishFormationSimulator) and follow the local instructions.
