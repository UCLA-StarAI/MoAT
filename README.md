# MoAT: Mixtures of All Trees
This repository contains the implementation of the MoAT model presented in our AISTATS23 paper titled `Mixtures of All Trees`. MoAT is a mixture over all possible $n^{(n-2)}$ tree-shaped graphical models over $n$ variables.

## Usage
- Install dependencies from `requirements.txt`.
- Run `./learn.sh` with the appropriate dataset name and hyperparameters of your choosing.

## Notes
- The collection of datasets used in our density estimation experiments is provided in the `datasets` folder. You may import your own datasets too. The current implementation requires that the datasets be binarized.
- The source code for our sampling experiments is availble in `sample.py`, and the experiments can be conveniently invoked with `./sample.sh`. Further, the methods in the `MoAT` class in `models.py` can be leveraged to obtain the samples directly too.