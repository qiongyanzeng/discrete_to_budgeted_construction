# Constructing Data-Driven Budgeted Uncertainty Sets for Robust Optimization

Code and data for the paper "Constructing Data-Driven Budgeted Uncertainty Sets for Robust Optimization" by Jamie Fairbrother, Marc Goerigk, and Qiongyan Zeng.

**experiment1.py**: Experiment 1: test the relationship between the approximation guarantee.

**experiment2.py**: Experiment 2: compare the performance of the three heuristics IH, OH, and SH to the exact algorithm IE.

**experiment3.py**: Experiment 3: compare IH, OH and SH results on large-scale instances.

**experiment4.py**: Experiment 4: bayesian optimization based on IH and OH solutions.

**experiment5.py**: Experiment 5: test and compare IH, OH and SH on Chicago travel time dataset.

**experiment6.py**: A.2. Experiment: employ the extreme-points approach introduced in Section 4.1 to approximate a budgeted uncertainty set by a discrete uncertainty set.

**Chicago_main.csv**: Chicago instances used in the experiments.

**Requirements:**
- Python 3.12.3
- Gurobi 12.0.2 (License required)
- Dependencies: See **requirements.txt**



## Usage

To run an experiment, execute the corresponding script, for example:

```
python experiment1.py 
```
