# README

## Project Overview

This project introduces **MTC-GA** (Multitasking Temporal Convolutional Network with Gated Recurrent Units and Attention), a novel multitasking framework for blood glucose forecasting. Designed to address the limitations of personalized and generalized approaches, MTC-GA strikes a balance between accuracy and computational efficiency. The model integrates task-specific and shared layers to enhance inter-patient knowledge transfer while retaining individual-specific patterns. Key contributions include:

- **Multitasking Architecture**: Combines TCN, GRU, and multi-head attention to provide personalized and generalizable predictions.
- **Efficient Training**: Reduces complexity by 40% compared to personalized models, enabling real-world applicability.
- **Wide Prediction Horizons**: Evaluates performance over multiple horizons (15-120 minutes).
- **Clinical Relevance**: Demonstrates low numerical and clinical errors for robust diabetes management.

---

## Project Structure

```
├── models
│   ├── generalized_model.py
│   ├── personalized_model.py
│   ├── shared_layer.py
│   ├── temporal_block.py
│   ├── helper_functions.py
├── src
│   ├── data_preprocessing.py
│   ├── dataset_class.py
│   ├── global_configs.py
│   ├── ohio_xml_converter.py
│   ├── post_processing.py
│   ├── time_series_prep.py
│   ├── visualizations.py
├── train-shared-layer-attention.ipynb
├── train-generalized-attention.ipynb
├── train-personalized-attention.ipynb
├── evaluate-approaches.ipynb
├── analysis.ipynb
├── requirements.txt
├── saved_models
│   ├── model_1.pth
│   ├── model_2.pth
│   ├── history_model_1.csv
│   ├── history_model_2.csv
```

### Key Components

- **`models Folder`**:

  - **`generalized_model.py`**: Implements the `GeneralizedModelWithAttention`, a TCN-GRU-Attention based forecasting model.
  - **`personalized_model.py`**: Implements the `PersonalizedModelWithAttention`, designed for individual forecasting tasks with similar architecture to the generalized model.
  - **`shared_layer.py`**: Implements the `SharedLayerModelWithAttention`, designed for multitask forecasting using shared and personalized layers.
  - **`temporal_block.py`**: Contains the implementation of `TemporalBlock`, a foundational component for Temporal Convolutional Networks (TCN).
  - **`helper_functions.py`**: Includes utilities for training, validating, and evaluating models.

- **`src Folder`** :

  - **`data_preprocessing.py`**: Provides utilities for preprocessing raw glucose data.
  - **`dataset_class.py`**: Implements the `TimeSeriesDataset` class for managing time series data.
  - **`global_configs.py`**: Contains global configurations such as batch size, device settings, and file paths.
  - **`ohio_xml_converter.py`**: Converts OhioT1DM XML files to CSV format.
  - **`post_processing.py`**: Provides utilities for analyzing model performance, including parameter counting.
  - **`time_series_prep.py`**: Functions for preparing data, aggregating patient data, and dataset splitting.
  - **`visualizations.py`**: Tools for plotting loss curves and performance metrics.

- **Saved Models**:

  - Stores trained models (`.pth` files) and corresponding histories in CSV format for easy reproducibility.

- **Notebooks**:

  - **`train-shared-layer-attention.ipynb`**: Training the shared layer attention model.
  - **`train-generalized-attention.ipynb`**: Training the generalized attention model.
  - **`train-personalized-attention.ipynb`**: Training the personalized attention model.
  - **`evaluate-approaches.ipynb`**: Comparing and evaluating forecasting models.
  - **`analysis.ipynb`**: Exploratory data analysis and performance evaluation.

- **`requirements.txt`**: Lists all dependencies for setting up the environment.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- tqdm
- NumPy
- scikit-learn
- Matplotlib

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To train a model, use the provided notebooks:

- **Generalized Attention Model**: Open and run `train-generalized-attention.ipynb`.
- **Personalized Attention Model**: Open and run `train-personalized-attention.ipynb`.
- **Shared Layer Attention Model**: Open and run `train-shared-layer-attention.ipynb`.

For evaluation and analysis, use:

- **Evaluation**: Open and run `evaluate-approaches.ipynb`.
- **Analysis**: Open and run `analysis.ipynb`.

---

