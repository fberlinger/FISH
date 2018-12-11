# BlueSim

Bla bla Project Explanation

## Additional Requirement if Animations are Desired

- ipyvolume

## Installation

Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload Code for an Experiment on the Virtual BlueBots

Go to the subfolder `fishfood`, choose one of the following experiments, and copy its file ending in `.py` to the current `BlueSim` folder:

- `orbit.py`: A single robot orbits around a fixed center.
- `millingabout.py`: Several robots orbit around a fixed center.
- `waltz.py`: Two robots orbit around each other.

Rename that file to `fish.py`.

## Run an Experiment with Simulated BlueBots

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open the file ending in `.ipynb` corresponding to your chosen experiment.

Please run each cell individually! **Warning**: Using `Run All` will not work
as the experiments start several threads for every fish and the code execution
async, hence, Jupyter Notebook runs new cells to quickly before others finished.

Do not run any cells generating the animation if you have not installed ipyvolume.

Sit back and watch the extravaganza!

<!---
## Run

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open one of the following experiment files ending in `.ipynb`:

- `millingabout.ipynb`
-->
