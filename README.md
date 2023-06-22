## Unbiased Pairwise Learning from Biased Implicit Feedback

---

### About

This repository accompanies the paper: A Replication Study: Unbiased Pairwise Learning from Biased Implicit Feedback
from Maxime Dassen, Abhijith Chintam and Ilse Feenstra

<!-- If you find this code useful in your research then please cite:

```
@
``` -->

### Datasets
To run the simulation with real-world datasets, the following datasets need to be prepared as described below.

- download the [Yahoo! R3 dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r) and put `train.txt` and `test.txt` files into `./data/yahoo/raw/` directory.
- download the [Coat dataset](https://www.cs.cornell.edu/~schnabts/mnar/) and put `train.ascii` and `test.ascii` files into `./data/coat/raw/` directory.

### Running the code
First, install the dependencies by running:

```bash
conda env create -f environment.yml
```
To preprocess the datasets, navigate to the `src/` directory and run the command

```bash
python preprocess_datasets.py -d coat yahoo
```

Then, run the following command in the same directory to reproduce the results from the paper: Unbiased Pairwise Learning from Biased Implicit Feedback By Saito

```bash
python run.py -d  coat -m wmf expomf relmf bpr ubpr ip --pointwise_loss paper_loss --pairwise_loss paper_loss
```

To reproduce the results for the methods DUMF and DUBPR, run: 

```bash
srun python run.py -d  coat -m dumf dubpr --pointwise_loss dual_unbiased_loss --pairwise_loss dual_unbiased_loss
```

This will run real-world experiments conducted in Section 4.

After running the experiments, you can summarize the results by running the following command in the `src/` directory.

```bash
python summarize_results.py -d yahoo coat
```

Once the code is finished executing, you can find the summarized results in `./paper_results/` directory.


### Acknowledgement

We want to thank Shashank Gupta for his helpful comments, discussions, and advice.

