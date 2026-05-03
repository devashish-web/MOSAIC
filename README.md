# Reproducing Experiments

## Environment Requirements

The experiments in this repository were run with the following environment:

- Python 3.11.7
- PyTorch 2.5.1+cu124
- Torch Geometric 2.7.0
- Scikit-learn 1.7.1
- NumPy 1.26.4

## Running Main Experiments

To reproduce the reported results, run:

```bash
python main.py --fed_algorithm MOSAIC --dataset cora --seed 0
```

The `--fed_algorithm` argument can be changed to run other baseline methods. The available algorithms are implemented in the `algorithm/` folder.

The `--dataset` argument can also be changed depending on the dataset. For example, the command above runs MOSAIC on the Cora dataset.

The reported results in the paper are obtained using five random seeds:

```bash
--seed 0
--seed 1
--seed 2
--seed 3
--seed 4
```

For example, to run MOSAIC on Cora with all five seeds:

```bash
python main.py --fed_algorithm MOSAIC --dataset cora --seed 0
python main.py --fed_algorithm MOSAIC --dataset cora --seed 1
python main.py --fed_algorithm MOSAIC --dataset cora --seed 2
python main.py --fed_algorithm MOSAIC --dataset cora --seed 3
python main.py --fed_algorithm MOSAIC --dataset cora --seed 4
```

## Running Baselines

### OASIS

To run the OASIS baseline, please refer to the official GitHub repository: [OASIS]([https://www.kaggle.com/datasets/alaaelmor/ton-iot-train-test-network](https://github.com/JiaruQian/OASIS))

For OASIS, the following dataset-specific hyperparameter values were used:

```python
# ----------------- Cora -----------------
parser.add_argument("--lambda_d", type=float, default=0.1)
parser.add_argument("--lambda_f", type=float, default=0.2)

# ----------------- Citeseer -----------------
parser.add_argument("--lambda_d", type=float, default=0.5)
parser.add_argument("--lambda_f", type=float, default=0.1)

# ----------------- Chameleon -----------------
parser.add_argument("--lambda_d", type=float, default=0.01)
parser.add_argument("--lambda_f", type=float, default=0.1)

# ----------------- Photo -----------------
parser.add_argument("--lambda_d", type=float, default=0.05)
parser.add_argument("--lambda_f", type=float, default=1.0)

# ----------------- Pubmed -----------------
parser.add_argument("--lambda_d", type=float, default=0.05)
parser.add_argument("--lambda_f", type=float, default=0.1)

# ----------------- OGBN-Arxiv -----------------
parser.add_argument("--lambda_d", type=float, default=0.5)
parser.add_argument("--lambda_f", type=float, default=0.1)
```

Please comment or uncomment the corresponding values according to the dataset being used.

### GHOST

For GHOST, the dataset-specific arguments can be modified in the `args.py` file. Please comment or uncomment the relevant settings according to the dataset.

## Running Ablation Studies

To run the MOSAIC ablation study, use:

```bash
python main.py --fed_algorithm Ablation --dataset cora --seed 0
```

The following ablation options are available.

### Atlas Ablation

```python
parser.add_argument(
    "--mosaic_atlas_ablation",
    type=str,
    default="none",
    choices=["none", "no_t"]
)
```

Example:

```bash
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_atlas_ablation no_t
```

### Graph Ablation

```python
parser.add_argument(
    "--mosaic_graph_ablation",
    type=str,
    default="none",
    choices=["none", "transition_only", "knn_only"]
)
```

Examples:

```bash
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_graph_ablation transition_only
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_graph_ablation knn_only
```

### Teacher Ablation

```python
parser.add_argument(
    "--mosaic_teacher_ablation",
    type=str,
    default="none",
    choices=["none", "uniform_average"]
)
```

Example:

```bash
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_teacher_ablation uniform_average
```

### Distillation Ablation

```python
parser.add_argument(
    "--mosaic_distill_ablation",
    type=str,
    default="full",
    choices=["full", "no_kd", "no_ce"]
)
```

Examples:

```bash
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_distill_ablation no_kd
python main.py --fed_algorithm Ablation --dataset cora --seed 0 --mosaic_distill_ablation no_ce
```

Each ablation setting should also be run using the following seeds:

```bash
--seed 0
--seed 1
--seed 2
--seed 3
--seed 4
```

## Running Non-Adaptive MOSAIC

To run Non-Adaptive MOSAIC, use:

```bash
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 0
```

For example, to run Non-Adaptive MOSAIC on Cora with all five seeds:

```bash
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 0
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 1
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 2
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 3
python main.py --fed_algorithm Non_Adaptive_MOSAIC --dataset cora --seed 4
```

## Notes

- Change `--dataset cora` to run experiments on another dataset.
- Change `--fed_algorithm MOSAIC` to run another baseline implemented in the `algorithm/` folder.
- All reported results should be reproduced using seeds `0, 1, 2, 3, 4`.
- For OASIS and GHOST, dataset-specific hyperparameters should be selected by commenting or uncommenting the corresponding settings in their configuration files.
