# Handwritten Digit Recognition (Neural Network)

Small learning project. The notebook `dl_project2.ipynb` trains a Keras Sequential network (Dense layers).

Data: Handwritten digit images, 10 classes, loaded in the notebook with a load_data call. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`src/benchmark.py` compares softmax regression, a 50-50 MLP and a 128-64 MLP with 5-fold cross-validation.
It uses scikit-learn's built-in digits dataset (1797 samples, 8x8 images).

## How to run

```
pip install numpy scikit-learn tensorflow jupyter
jupyter notebook
python src/benchmark.py
```


## Results (5-fold CV, mean)

| Model | accuracy | macro_f1 | log_loss |
|---|---|---|---|
| Deep MLP (128-64 ReLU + Adam + L2) | 0.9839 | 0.9838 | 0.0608 |
| Two-Layer MLP (Notebook 50-50 ReLU) | 0.9789 | 0.9788 | 0.0724 |
| Multinomial Softmax Baseline | 0.9633 | 0.9634 | 0.1929 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `dl_project2.ipynb`: notebook
- `src/benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
