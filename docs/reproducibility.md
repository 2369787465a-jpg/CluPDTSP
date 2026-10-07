# Reproducing the numerical pipeline

This guide describes how to generate new experiment artifacts from the published source. The [README](../README.md) provides installation instructions, a small training example, and the manuscript link.

## Release contents and limits

| Artifact | Available in this source release |
| --- | --- |
| CAADRL model, environment, training, and evaluation code | Yes |
| Heter baseline source, upstream license, and usage instructions | Yes |
| Dataset generators, statistics, route diagnostics, and profiling scripts | Yes |
| Manuscript PDF, available lightweight summaries, and figures | Yes |
| Available historical Heter training configurations (`args.json`) | Yes |
| Pretrained weights, pregenerated test data, and instance-level raw results | No |
| Full experiment logs and private execution audits | No |

The local source archive contained Git LFS pointers for the omitted large artifacts. A pointer is a small metadata file, not the dataset, checkpoint, or CSV it names. This release does not publish those pointers or require LFS. Paths in retained summaries refer to historical experiment inputs and outputs; they are provenance references rather than bundled files.

The full training and benchmark results in the manuscript have not been rerun for this source release. The release was checked with Python syntax compilation, deterministic synthetic data generation, and a small CPU model/environment rollout that verifies complete customer visits, pickup-before-delivery constraints, finite objectives, and closed-tour length.

## Environment

Install `requirements.txt` in a Python 3.9+ environment. Full training and large sampling widths are intended for CUDA GPUs; the principal CAADRL entry points also support CPU execution. Match the PyTorch build to the machine and record the Python, PyTorch, CUDA, driver, GPU, and CPU versions for any new benchmark.

Reported runtimes depend on hardware, batch size, decoding width, and synchronization. Compare runtimes only under a documented common measurement setup.

## Generate shared test sets

The number of customer nodes excludes the depot and must be even. For every instance, the first half of the customers are pickups and the second half are their corresponding deliveries.

Run from the repository root:

```bash
python scripts/build_unified_test_sets.py --sizes 10 20 40 80 --distributions clustered uniform --num-instances 10000 --seed 10000 --cluster-std 0.1
```

The output directory defaults to `data/pdp/unified/`. For example, the clustered PDP40 output is `pdp40_test_clustered_std0.1_seed10000_n10000.pkl`. Instance `i` uses seed `10000 + i`. Preserve the same files and instance order for every method being compared. Add `--force` to intentionally regenerate existing files.

The aggregate datasets created here are consumed by `scripts/evaluate_caadrl.py` and `scripts/evaluate_heter.py`. The simple `experiment.py` entry point instead reads one instance from each individual file produced by `generate_pdp_dataset.py`.

## Train CAADRL

For a clustered PDP40 experiment:

```bash
python run_full.py --task train --problem_size 40 --distribution clustered --seed 1234 --epochs 800 --train_batch_size 128 --result_dir result/full
```

This saves model checkpoints under `result/full/_train_pdtsp_n40_clustered_full_gate/`. The final checkpoint is `checkpoint-800.pt`. Training instances are generated on the fly.

Use the wrappers documented in the README for ablations. Keep the problem size, training budget, distribution, seed policy, and evaluation data comparable. A trained checkpoint must be evaluated with the same model architecture and ablation settings.

## Evaluate CAADRL on an aggregate dataset

After generating the test set and training the checkpoint:

```bash
python scripts/evaluate_caadrl.py --dataset data/pdp/unified/pdp40_test_clustered_std0.1_seed10000_n10000.pkl --checkpoint result/full/_train_pdtsp_n40_clustered_full_gate/checkpoint-800.pt --output results/raw/caadrl/pdp40_clustered_greedy.csv --ablation full --size 40 --distribution clustered --decode greedy --seed 10000 --num-instances 10000
```

For sampling, change `--decode greedy` to `--decode sampling --width 1280` and choose a different output name. `--no-cuda` forces CPU execution, and `--batch-size` controls evaluation batches. The raw CSV records instance identifiers, objective, route, runtime, checkpoint path, and invocation.

The environment stores the initial depot and each customer in the selected sequence; the distance calculation includes the final return edge to the depot.

## Train and evaluate the Heter baseline

Follow [the retained upstream instructions](../baselines/heter/README.md) to train a size-matched Heter model. Its checkpoint format differs from CAADRL's. Supply the path to the resulting Heter checkpoint to this evaluation command:

```bash
python scripts/evaluate_heter.py --dataset data/pdp/unified/pdp40_test_clustered_std0.1_seed10000_n10000.pkl --model path/to/heter_checkpoint.pt --output results/raw/heter/pdp40_clustered_greedy.csv --size 40 --distribution clustered --decode greedy --seed 10000 --val-size 10000
```

Use the same dataset and decoding budget for both methods. Existing `checkpoints/heter/**/args.json` files are historical configurations, not pretrained models; machine-specific paths inside them must be adapted for a new run.

## Validate and summarize new results

Validate a newly generated CAADRL CSV:

```bash
python scripts/validate_raw_results.py --raw results/raw/caadrl/pdp40_clustered_greedy.csv --expected-n 10000 --size 40 --distribution clustered --decode Greedy --seed 10000
```

Repeat for the baseline file. The validator checks the required schema, row count, contiguous instance identifiers, metadata, finite objectives and runtimes, provenance fields, and route JSON.

Create aggregate tables:

```bash
python scripts/summarize_raw_results.py "results/raw/caadrl/*.csv" --csv-output results/summary/new_caadrl_results.csv --tex-output results/summary/new_caadrl_results.tex
```

Compute paired statistics only after both raw CSV files exist and refer to exactly the same dataset, instance IDs, size, distribution, decoding label, and seed:

```bash
python scripts/paired_statistics.py --caadrl results/raw/caadrl/pdp40_clustered_greedy.csv --baseline results/raw/heter/pdp40_clustered_greedy.csv --baseline-label Heter --output results/stats/new_caadrl_vs_heter.csv --bootstrap 1000 --seed 1234
```

The output includes mean paired percentage gap, a bootstrap 95% confidence interval, a one-sample t-test, a standardized effect size, and win/tie/loss counts. Negative paired gaps mean lower CAADRL tour length. Do not infer instance-level paired statistics from summary means alone.

## Additional analyses

Use the relevant script's `--help` to configure these optional interfaces:

- `compute_route_metrics.py`, `summarize_route_metrics.py`, and `plot_route_comparison.py`: route mechanisms and visual comparisons.
- `profile_heter.py` and `summarize_profiling.py`: warm-up, synchronized inference timing, and memory statistics.
- `convert_li_lim_to_pdp_relaxed.py`: external benchmark conversion, requiring separately obtained original instances.
- `run_ablation_grid.sh` or `run_ablation_grid.ps1`: larger ablation jobs, requiring a suitable local compute environment.

The Li-Lim conversion removes time windows, capacity, and the original multi-vehicle objective. Its resulting Euclidean single-vehicle pickup-delivery tours should therefore be assessed as a relaxed routing benchmark, without claiming direct comparability to original PDPTW best-known solutions.

New datasets, checkpoints, raw results, and logs are ignored by Git by default. Preserve them separately with their configurations and hardware records when conducting new experiments.
