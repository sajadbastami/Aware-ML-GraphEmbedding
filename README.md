# Aware-ML-GraphEmbedding
# An Aware-Feature Method for Node Classification on Complex Graphs with Multi-Label Datasets

This repository provides an implementation and experimental summary of the paper:

**“An Aware-Feature Method for Node Classification on Complex Graph with Multi-Label Dataset”**

## Overview
This paper proposes a graph representation learning method designed specifically for **multi-label attributed networks**.  
Unlike many existing approaches that assume each node has only a single label, this method explicitly models **label correlations** and integrates them into the embedding process.

The proposed framework combines:

- **Random walk-based neighborhood sampling**
- **SimHash-based label encoding**
- **Word2Vec-based feature representation**
- **CNN-based classification**

to improve node classification on complex graph data.

## Problem Statement
Many existing graph embedding methods are effective only for **single-label graphs** and do not fully capture the semantics of **multi-label networks**.  
This work addresses the challenge of learning node representations that preserve:

- **Node content**
- **Neighborhood structure**
- **Label correlation information**

## Proposed Method
The proposed pipeline contains **three main stages**:

1. **Feature Extraction**  
   Neighboring node labels are collected using random walk sampling to preserve local graph structure and proximity.

2. **Node Image Generation**  
   - Labels are encoded using **SimHash**
   - Encoded values are converted into **Word2Vec image representations**

3. **CNN Training**  
   The generated representations are used to train a **Convolutional Neural Network (CNN)** for node classification.

### Key Design Principles
- Preserves **first-order proximity**
- Preserves **second-order proximity**
- Uses neighborhood labels to uncover **latent label correlations**
- Combines structural and semantic information in one embedding pipeline

## Contributions
- A **multi-label graph embedding method** for node classification
- A **label-correlation-aware** representation learning strategy
- A pipeline based on **Random Walk + SimHash + Word2Vec + CNN**
- Improved handling of **complex attributed graphs**

## Datasets
The paper evaluates the method on the following benchmark datasets:

- **Flickr**
- **BlogCatalog**
- **Citeseer**
- **Cora**

### Dataset Characteristics
- **Flickr**: 89,250 nodes, 899,756 edges
- **BlogCatalog**: 10,312 vertices, 333,982 edges, 39 labels
- **Citeseer**: 3,327 papers, 4,732 links, 6 classes, 3,703-dimensional binary features
- **Cora**: 2,708 papers, 5,429 links, 7 classes, 1,433-dimensional binary features

## Evaluation Metrics
The paper evaluates the proposed method using:

- **Micro-F1**
- **Hamming Loss**
- **Accuracy**

## Baseline Methods
The method is compared against several strong baselines, including:

- Node2Vec
- DeepWalk
- LINE
- MLNE
- MGCN
- Planetoid
- GCN
- GAT
- Attention Walk
- HOPE
- MLGW
- LANC

## Reported Results
The paper reports improvements over multiple baseline methods on the benchmark datasets.

The comparison tables include:

- **Micro-F1 comparison**
- **Hamming loss comparison**
- **Node classification accuracy comparison**

> Note: The extracted content clearly shows the evaluation tables and metrics, but the exact numeric values were not fully visible in the extracted text.

## Repository Structure
```text
.
├── data/                  # Dataset files
├── src/                   # Core implementation
├── models/                # CNN / embedding models
├── notebooks/             # Experiments and analysis
├── results/               # Outputs, plots, and tables
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Recommended Python Version

    Python 3.9
    Python 3.10

Required Libraries and Versions

The project has been tested with the following package versions:

    numpy==1.23.5
    scipy==1.10.1
    scikit-learn==1.2.2
    networkx==3.1
    gensim==4.3.1
    pandas==1.5.3
    matplotlib==3.7.1
    tensorflow==2.12.0

Usage

After installing dependencies, you can run the pipeline in the following order:

    Prepare the dataset
    Extract node neighborhoods using random walk
    Encode labels using SimHash
    Generate word2vec-based representations
    Train the CNN classifier
    Evaluate using Micro-F1, Hamming Loss, and Accuracy
