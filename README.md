A comparative study of Support Vector Machines (SVM) and Graph Convolutional Networks (GCN) for detecting fake reviews, featuring statistical validation through repeated experiments.

## Project Structure
```bash
.
├── dataset/
│   ├── fake_reviews_dataset.csv      # Raw dataset
│   └── fake_reviews_cleaned.csv      # Preprocessed dataset
├── utility/
│   ├── gcn_util.py           # GCN model implementation
│   └── svm_util.py           # SVM model implementation
├── 1_Prepocessing.ipynb               # Data cleaning pipeline
├── 2B_GCN_Model.ipynb                  # GCN model development
├── 2A_SVM_Model.ipynb                  # SVM model development
├── 3_Experiment.ipynb                 # Comparative analysis pipeline
└── README.md                          # This documentation