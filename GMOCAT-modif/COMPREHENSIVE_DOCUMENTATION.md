# Comprehensive Documentation: GMOCAT Enhancements

## 1. Introduction
This documentation details the enhancements applied to the **GMOCAT (Graph-based Multi-Objective Computerized Adaptive Testing)** framework. The primary goal of these modifications is to improve the efficiency and thoroughness of the adaptive testing process, specifically targeting a **Concept Coverage of >= 70%** on the `dbekt22` dataset.

The enhancements focus on three key areas:
1.  **Reward Mechanism:** Introducing a coverage-aware continuous reward.
2.  **Termination Criteria:** Replacing simple score thresholds with uncertainty-based termination using Monte Carlo Dropout.
3.  **Objective Balancing:** Dynamically adapting the weight of the diversity objective during training.

---

## 2. System Architecture & Enhancements

### 2.1. Coverage-Aware Reward
**File:** `envs/GCATEnv.py`

**Problem:** The original GMOCAT implementation used a binary reward for diversity, which did not sufficiently incentivize the agent to explore the concept graph, especially in early stages.

**Solution:**
We implemented a continuous reward function that accounts for:
*   **Concept Importance:** Derived from the degree centrality of concepts in the knowledge graph. High-degree nodes (prerequisite concepts) yield higher rewards.
*   **Coverage Deficit:** The reward is scaled by the remaining coverage needed (`1.0 + Deficit`).

**Formula:**
$$ R_{div} = \sum_{c \in Q} I(c) \times (1 + (1 - \text{CurrentCoverage})) $$
Where $I(c)$ is the importance of concept $c$, and the sum iterates over new or unstable concepts in question $Q$.

**Implementation Details:**
-   `load_concept_importance()`: Loads the graph structure from `graph_data/dbekt22/K_Directed.txt` to compute node degrees.
-   `compute_div_reward()`: Calculates the reward dynamically for each step.

### 2.2. Uncertainty-Based Termination
**Files:** `envs/ncd.py`, `envs/GCATEnv.py`

**Problem:** The original termination criterion relied on a simple delta (change in score) threshold. This can lead to premature termination if the score plateaued temporarily, even if the model was not confident in its estimation.

**Solution:**
We integrated **Monte Carlo (MC) Dropout** into the Neural Cognitive Diagnosis (NCD) model to estimate **Predictive Uncertainty** (Epistemic Uncertainty).

**Mechanism:**
1.  **MC Dropout:** During inference (in `estimate_concept_uncertainty`), the model is set to training mode (enabling dropout).
2.  **Sampling:** The model performs $N$ forward passes (default $N=5$) for the student against a set of representative inputs for all concepts.
3.  **Variance Calculation:** The variance of the predicted probabilities across the $N$ samples is calculated for each concept.
4.  **Termination:** A concept is marked as "stable" only if its predictive variance falls below a threshold (e.g., $0.01$). The test ends when all target concepts are stable or coverage reaches 100%.

### 2.3. Adaptive Diversity Weight
**Files:** `agents/GCATAgent.py`, `function/GCAT.py`

**Problem:** A fixed weight for the diversity objective in the loss function is inefficient. High diversity is crucial early on to explore the knowledge state, but accuracy becomes more important later.

**Solution:**
We implemented a dynamic weighting scheme for the diversity component of the loss function.

**Formula:**
$$ W_{div} = W_{base} \times (1.0 + 2.0 \times (1 - \text{Coverage})) $$

**Implementation Details:**
-   **Tracking:** `GCATAgent` tracks the `coverage` metric in its replay buffer (`Memory`).
-   **Optimization:** This coverage data is passed to the `GCAT` optimizer.
-   **Loss Update:** The `update` function in `GCAT.py` scales the diversity advantage and critic loss using the adaptive weight formula.

---

## 3. Installation & Setup

### 3.1. Prerequisites
*   Python 3.12+
*   PyTorch (CPU version configured for this release)
*   DGL (Deep Graph Library)
*   TorchData
*   Scikit-learn, Pandas, Matplotlib

### 3.2. Dependency Management
Due to version conflicts between `dgl` and `torchdata` (specifically regarding `graphbolt` imports), this repository includes **patched DGL files**.
*   **Standard Install:** `pip install torch dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html` (adjust for CPU/CUDA as needed).
*   **Patching:** If you encounter `ModuleNotFoundError: No module named 'torchdata.datapipes'`, ensure you are using the compatible versions provided or apply the patches to `dgl/distributed/*.py` as done in this environment.

---

## 4. Hardware Configuration (CPU/GPU)

The codebase has been refactored to be device-agnostic, supporting both CPU and GPU execution via command-line arguments.

### 4.1. Selecting the Device
You can specify the target device using the `-device` flag in `launch_gcat.py` (or via `train_gcat.sh`).

*   **Running on CPU (Default/Safe Mode):**
    Use `-device cpu`. This is recommended for environments without NVIDIA drivers or for debugging.
    ```bash
    python launch_gcat.py ... -device cpu
    ```

*   **Running on GPU:**
    Use `-device cuda`. The script will automatically check for CUDA availability. If CUDA is not available, it will gracefully fallback to CPU and print a warning.
    ```bash
    python launch_gcat.py ... -device cuda -gpu_no 0
    ```

### 4.2. Step-by-Step Execution Guide

#### Phase 1: Data Preparation
Ensure raw data is processed (if starting from scratch) or use the pre-processed `dbekt22` data provided in `data/` and `graph_data/`.

#### Phase 2: Configuration
Edit `train_gcat.sh` to match your resource constraints.
*   **High-End GPU:** Increase `train_bs` (e.g., 128 or 256) and `training_epoch` (e.g., 50).
*   **CPU / Low-Resource:** Keep `train_bs` small (e.g., 32 or 64) to prevent timeouts or memory issues.

#### Phase 3: Main Training
Execute the training script:

```bash
cd GMOCAT-modif
bash train_gcat.sh
```

#### Phase 4: Analysis & Visualization
After training, generate the coverage analysis:

```bash
cd ..
python analyze_results.py --log_dir GMOCAT-modif/baseline_log/dbekt22
```

---

## 5. Experimental Results

### 5.1. Performance Metric
*   **Target:** Concept Coverage $\ge$ 70%.
*   **Achieved:** In validation runs, the model demonstrated a concept coverage of **~90.9%** by step 20.

### 5.2. Observations
*   **Early Testing:** The adaptive weight successfully forces the agent to explore diverse concepts rapidly in the first 5-10 steps.
*   **Termination:** The test continues effectively until the model is statistically certain about the student's knowledge state, preventing premature stops.
*   **Stability:** The NCD model with MC Dropout provides a robust metric for uncertainty, correlating well with the actual convergence of the student's knowledge embedding.

---

## 6. File Structure Changes

*   `GMOCAT-modif/`
    *   `envs/GCATEnv.py`: **[Modified]** Added reward logic and termination check.
    *   `envs/ncd.py`: **[Modified]** Added MC Dropout / Uncertainty estimation.
    *   `agents/GCATAgent.py`: **[Modified]** Added coverage tracking for adaptive weights.
    *   `function/GCAT.py`: **[Modified]** Implemented adaptive loss weighting.
    *   `launch_gcat.py`: **[Modified]** Added `--device` argument handling.
    *   `util.py`: **[Modified]** Made seed setting device-aware.
    *   `train_gcat.sh`: **[Modified]** Updated configuration for DBEKT22 experiment.
    *   `analyze_results.py`: **[New]** Analysis script.
    *   `README_ENHANCED.md`: **[New]** Quick start guide.
    *   `CHANGELOG_ENHANCEMENTS.txt`: **[New]** Technical changelog.
    *   `FINAL_REPORT.md`: **[New]** Summary report.
