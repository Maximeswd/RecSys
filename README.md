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
To preprocess the datasets, navigate to the `src/` directory and run the command. the -p command runs the preprocessing for multiple propensity estimations.

```bash
python preprocess_datasets.py -d coat yahoo -p original bb-item bb-item-user
```

Then, run the following command in the same directory to reproduce the results from the paper: Unbiased Pairwise Learning from Biased Implicit Feedback By Saito.

```bash
python run.py -d  coat -m wmf expomf relmf bpr ubpr ip --pointwise_loss original --pairwise_loss original -p original -r 10
```
To run the original models with different propensity estimations:

```bash
python run.py -d  coat -m wmf expomf relmf bpr ubpr ip --pointwise_loss original --pairwise_loss original -p bb-item bb-item-user -r 10
```

To reproduce the results for the methods DUMF and DUBPR with the different propensity estimations, run: 

```bash
python run.py -d  coat -m dumf dubpr --pointwise_loss dual_unbiased --pairwise_loss dual_unbiased -p original bb-item bb-item-user -r 10
```

After running the experiments, you can summarize the results by running the following command in the `src/` directory.

```bash
python summarize_results.py -d yahoo coat -p original bb-item bb-item-user
```

Once the code is finished executing, you can find the summarized results in `./paper_results/` directory.

### Running on cluster
Additionally, we made scripts to make the code executable on a cluster. Redirect to 'scripts/' directory and change the srun command to the desired preprocess step as described above.
The command to run the script is:
```bash
sbatch preprocess.job
```
Similarly, change the command after srun in run.job to run the experiments:
```bash
sbatch run.job
```
Finally, change the command after srun in summarize.job to summarize the experiments:
```bash
sbatch summarize.job
```


### Acknowledgement

We want to thank Shashank Gupta for his helpful comments, discussions, and advice.

