# Ablation Studies — OPE and RoPE

This folder contains the positional-encoding ablations for all transformer models in the
paper. Each ablation isolates the contribution of the orientation-aware positional
embedding (OPE) and, where applicable, Rotary Position Embeddings (RoPE).

Within each notebook, all variants share the same architecture, hyperparameters, data,
augmentation and training protocol. Only the positional-encoding configuration changes.

## Notebooks

| Ablation notebook | Main notebook | Variants |
|-------------------|---------------|----------|
| `A02_SimpleViT_linear_decoder_ablation.ipynb` | `../02_simpleViT_linear_decoder.ipynb` | `baseline`, `OPE_only` |
| `A03_SimpleViT_attention_decoder_ablation.ipynb` | `../03_simpleViT_attention_decoder.ipynb` | `baseline`, `OPE_only` |
| `A04_TransUNet_B32_ablation.ipynb` | `../04_TransUNet_B32.ipynb` | `baseline`, `OPE_only` |
| `A05_TransUNet_B32_RoPE_ablation.ipynb` | `../05_TransUNet_B32_with_RoPE.ipynb` | `baseline`, `OPE_only`, `rope_only`, `full` |
| `A06_TransUNet_RES_RoPE_ablation.ipynb` | `../06_TransUNet_RES_RoPE.ipynb` | `baseline`, `OPE_only`, `rope_only`, `full` |

## Variants

| `VARIANT` | OPE | RoPE | Description |
|-----------|:---:|:----:|-------------|
| `baseline`  | ✗ | ✗ | The model's built-in positional embedding only |
| `OPE_only`  | ✓ | ✗ | Built-in embedding + OPE |
| `rope_only` | ✗ | ✓ | Built-in embedding + RoPE in the encoder attention |
| `full`      | ✓ | ✓ | Built-in embedding + OPE + RoPE |

Note that `baseline` is not "no positional information": it keeps the positional embedding
that the backbone already has (the 2D sin-cos embedding of SimpleViT, or the learned
embedding of TransUNet). Each variant adds components on top of it.

## Required Data

The ablation notebooks build their data modules from the preprocessed cache, not from the
raw `data_sets/` folders. Before running, make sure these files exist in Google Drive:

```
/content/drive/MyDrive/TranSpeckle/cache/
├── 2400/processed.pt
├── 1600_4/processed.pt
├── 1600_2/processed.pt
├── 800_4/processed.pt
├── 800_2/processed.pt
├── 800_1/processed.pt
├── group1/processed.pt
├── group2/processed.pt
├── group3/processed.pt
├── group4_1/processed.pt
└── group4_2/processed.pt
```

This is the same cache used by the main notebooks (see the main [README](../../README.md)).
If a cache file is missing, build it from `data_sets/` with the preprocessing cells of the
corresponding main notebook.

## How to Run

Each variant is trained in a **separate run**. One run trains the selected variant on all
six training sets (`2400`, `1600_4`, `1600_2`, `800_4`, `800_2`, `800_1`).

1. Open the ablation notebook in Google Colab (use the "Open in Colab" badge).
2. In the configuration cell near the top, set the variant:
   ```python
   VARIANT = "baseline"   # see the Variants column of the table above
   ```
3. Run all cells (**Runtime → Run all**). The notebook trains the variant on each training
   set and saves the checkpoints.
4. When the run finishes, change `VARIANT` to the next variant and run again.
   Repeat until every variant listed for that notebook has been run.
5. After all variants are done, run the **last two cells** of the notebook once:
   - The first evaluates every trained variant on `group1` and saves the results to a CSV file.
   - The second prints the ablation tables (each variant vs. `baseline`, IoU and PCC) and
     a reproduction check of the re-trained model against the numbers reported in the paper.

## Outputs

Each run writes to a separate directory named after the model and the variant, so runs
do not overwrite each other:

```
/content/drive/MyDrive/TranSpeckle/optuna_logs/<model_name>/<model_name>/fixed_runs/<model_name>_<train_set>/
├── checkpoints/        # best_*.ckpt and last_*.ckpt
├── metrics.csv         # training log
└── run_config.json     # hyperparameters and variant flags used for the run
```

`<model_name>` is set automatically from the notebook and the variant
(for example `TransUNet_res_rope_only`).

The summary results are saved to:

| Notebooks | Results file |
|-----------|--------------|
| A02, A03 | `TranSpeckle/ablation_results_SimpleViT.csv` |
| A04, A05 | `TranSpeckle/ablation_results_B32.csv` |
| A06 | `TranSpeckle/ablation_results.csv` |

## Notes

- **No trained weights are provided for the ablations.** Each run trains the selected
  variant from scratch. The R50-ViT notebook (A06) downloads the ImageNet-21k R50+ViT-B/16
  backbone weights automatically, as in the main notebook.
- Data paths, the Google Drive layout and the preprocessed cache are the same as for the
  main notebooks. See the main [README](../../README.md) for the required Drive structure.
  All variants load the same cached data.
- Each run trains six models, so a full run is long. A Colab Pro runtime (e.g., A100) is
  recommended; on the free tier a run may exceed the session time limit.
- All runs use a fixed random seed (`seed_value = 11`, set in the imports cell). To check
  how much the results vary across random seeds, change `seed_value` in the main notebooks
  and re-train. Use a different project name or output directory for each seed, so the new
  run does not overwrite the existing one.
- Results are reported in the ablation section of the paper.
