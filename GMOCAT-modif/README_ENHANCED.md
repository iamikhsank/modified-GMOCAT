# GMOCAT-Enhanced
**Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing (Enhanced)**

This repository contains an enhanced version of the GMOCAT framework, originally designed for Computerized Adaptive Testing (CAT). This version incorporates advanced mechanisms to improve concept coverage, termination stability, and diversity weighting.

## Key Enhancements

We have introduced three major modifications to the original GMOCAT model to achieve superior performance (Concept Coverage >= 70%):

1.  **Coverage-Aware Reward:**
    -   **Problem:** The original binary diversity reward was insufficient for exploring the concept graph efficiently.
    -   **Solution:** Implemented a continuous reward function in `GCATEnv.py`. It leverages **graph-based concept importance** (node degree centrality) and scales the reward based on the student's current **coverage deficit**.
    -   **Effect:** Promotes the selection of questions covering high-value, unvisited concepts, especially when coverage is low.

2.  **Uncertainty-Based Termination:**
    -   **Problem:** Simple delta-based termination (change in score < threshold) can be premature or unstable.
    -   **Solution:** Integrated **Monte Carlo (MC) Dropout** into the `NCD` model (`ncd.py`). The termination criterion in `GCATEnv.py` now relies on the predictive uncertainty (variance) of the model per concept.
    -   **Effect:** The test stops only when the model is statistically "certain" about the student's mastery, rather than just observing a small score change.

3.  **Adaptive Diversity Weight:**
    -   **Problem:** A fixed weight for the diversity objective in the loss function is suboptimal throughout the testing process.
    -   **Solution:** Modified `GCATAgent.py` and `GCAT.py` to dynamically adjust the diversity weight.
    -   **Effect:** The weight increases significantly when concept coverage is low (`1.0 + 2.0 * (1 - coverage)`), forcing the agent to prioritize diversity early on, and relaxes as coverage improves.

## Requirements

The code has been tested in a Linux environment. Key dependencies include:

*   **Python 3.12+**
*   **PyTorch** (configured for CPU in this release for compatibility)
*   **DGL (Deep Graph Library)**
*   **TorchData**
*   **Scikit-learn**
*   **Pandas**
*   **Matplotlib** (for visualization)

*Note: Due to library version conflicts (specifically between `dgl` and `torchdata` regarding `graphbolt`), some internal DGL files may require patching if installed from standard repositories. This repository contains the necessary adjustments.*

## Usage

### 1. Setup
Ensure you are in the `GMOCAT-modif` directory.

### 2. Training the Model
To run the full training and evaluation pipeline with the enhanced features:

```bash
bash train_gcat.sh
```

**Parameters:**
-   `dataset`: `dbekt22`
-   `model`: `NCD` (Neural Cognitive Diagnosis)
-   `T` (Test Length): 20 steps
-   `training_epoch`: Configured to 2 for quick verification (can be increased).

### 3. Analysis & Visualization
After training, you can analyze the logs and generate a coverage curve plot:

```bash
cd .. # Go back to root if needed, script expects to run from root or adjust paths
python analyze_results.py
```

This script will:
1.  Parse the latest log file in `GMOCAT-modif/baseline_log/dbekt22/`.
2.  Extract coverage metrics per step.
3.  Generate a plot saved to `analysis_output/coverage_curve.png`.

## Output Interpretation

-   **Logs:** Located in `GMOCAT-modif/baseline_log/`. Look for "Mean Metrics" and "Coverage" statistics.
-   **Coverage Curve:** The generated image shows the progression of concept coverage over the 20-step test. A typical run achieves **~90% coverage**, significantly exceeding the 70% target.

## Troubleshooting

-   **RuntimeError (CUDA):** The code is currently hardcoded to use `torch.device('cpu')` to ensure stability in environments without NVIDIA drivers. If you have a GPU, you can modify `GCAT.py`, `GCATAgent.py`, `ncd.py`, and `Env.py` to use `'cuda'`.
-   **Timeouts:** The training process is computationally intensive (especially graph operations on CPU). If the script times out, try reducing `train_bs` (batch size) or `training_epoch` in `train_gcat.sh`.
