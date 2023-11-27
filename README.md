# A Novel Online EEG-based Error Detection Approach Using Effect Size and Innovative Feature Types

## Project Description

With this study, we demonstrated a novel approach for online asynchronous detection of ErrPs, which emerged as the winner of a competition at the 32nd International Joint Conference on Artificial Intelligence. The competition challenged participating teams to develop and apply machine learning models for real-time error detection using EEG data from participants operating an active orthosis. The competition consisted of two stages: an offline stage for model development using pre-recorded, labelled EEG data, and an online stage three months after the offline stage, where these models were tested live on streamed, unlabelled EEG data to detect errors in orthosis movements in real time. Our approach employed innovative feature types and selection strategies, in particular the use of time-derivative EEG amplitude features, as well as Cohen's d values to identify the most discriminative and consistent features, integrating neurophysiological knowledge into a data-driven model, thereby enhancing the robustness and reliability of error detection. The model trained in the offline stage not only resulted in a high average cross-validation accuracy of 97.8% across all participants, but also demonstrated remarkable robustness against cross-day variability in EEG when applied in the online session three months later, achieving 79.2% accuracy without recalibration, while maintaining prompt response capabilities. Our research provides a novel approach for general event-related potential-based online classification tasks, with significant implications for the future of human-robot interaction in terms of robust and reliable error detection in practical, real-world scenarios.

## Contents of This Repository

This repository includes scripts utilized in the "Intrinsic Error Evaluation during Human-Robot Interaction" (IntEr-HRI Competition, https://ijcai-23.dfki-bremen.de/competitions/inter-hri/) at the 32nd International Joint Conference on Artificial Intelligence (IJCAI 2023, https://ijcai-23.org/):

1. Python scripts:
   - `classifier_training.py` for offline classifier training and simulating online error prediction with a moving window
   - `online_prediction.py` for real-time online error prediction.
   - `streaming_simulation.py` to simulate real-time EEG data streaming via LSL.

2. MATLAB scripts for classifier training across all 8 participants.

The dataset referenced in this study is available at: [https://zenodo.org/records/8345429]
Reference: Kueper, N., Chari, K., Bütefür, J., Habenicht, J., Kim, S. K., Rossol, T., Tabie, M., Kirchner, F., & Kirchner, E. A. (2023). EEG and EMG dataset for the detection of errors introduced by an active orthosis device. In arXiv. http://arxiv.org/abs/2305.11996


### Authors

- Yanzhao Pan
- Dr. Marius Klug  
Intuitive XR Labs, BTU Cottbus-Senftenberg, Germany

### Dependencies

- Python 3.8.8 (other libraries as per `requirements.txt`)
- MATLAB, EEGLAB

## Installation

Clone the repository and install dependencies:


