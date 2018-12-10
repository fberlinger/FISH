# BlueSim

Bla bla Project Explanation

## Additional Requirement if Animations are Desired

- ipyvolume

## Installation

Manually following instructions on https://github.com/maartenbreddels/ipyvolume.

## Upload code for an experiment on the virtual BlueBots

Go to the subfolder `fishfood` and copy one of the following files ending in `.py` to the current `BlueSim` folder:

- `millingabout.py`

Rename that file to `fish.py`.

## Run an experiment with simulated BlueBots

Open the jupyter notebook:

```
jupyter notebook
```

and within that notebook open the corresponding file ending in `.ipynb`.

Please run each cell individually! **Warning**: Using `Run All` will not work
as the experiments start several threads for every fish and the code execution
async, hence, Jupyter Notebook runs new cells to quickly before others finished.

Do not run any cells generating the animation if you have not installed ipyvolume.

Sit back and watch the extravaganza!
