This repository is part of the CMPE 255, HomeWork 2.

### Setup

```bash
conda create -n intel -c conda-forge python=3.9 scikit-learn-intelex
conda activate intel
```

### Files

- *benchmark.py* : main benchmarking script. To learn how to use it, run `python benchmark.py -h`
- *model.py* : Wrapper classes that run all the optimized regression and classification models.

### How to Use

First, clone the git repo and `cd` into it
```bash
git clone https://github.com/Ankur-singh/cmpe255
cd cmpe255
```

Next, run the `make` command as follows. Its can take anywhere around **30 to 50 mins**, depending on your system config.

**Note:** You don't have to run it all, you can find all the benchmarking stats inside *results* folder.

```bash
make -f Makefile all
```

The above commands will do the following:
 - `clean` : clean the *results* directory	
 - `regression` : Run Regression models without Intel Optimization
 - `classification` : Run Classification models without Intel Optimization
 - `regression_intel`: Run Regression models with Intel Optimization
 - `classification_intel` : Run Classification models with Intel Optimization

Please check *Makefile* for more details.
