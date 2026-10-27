# E-Commerce Customer Segmentation Using K-Means Clustering

A machine learning project that applies **K-Means clustering** to segment retail customers based on their annual income and spending scores. The project explores customer behaviour through exploratory data analysis, clustering, visualisation and evaluation.

## Overview

Understanding customer behaviour helps businesses develop targeted marketing strategies and improve customer engagement. This project uses unsupervised machine learning to identify groups of customers with similar characteristics based on their annual income and spending scores.

## Objectives

- Perform exploratory data analysis (EDA) on customer data.
- Visualise customer attributes and spending patterns.
- Preprocess and standardise numerical features.
- Use the Elbow Method to investigate the optimal number of clusters.
- Apply K-Means clustering to identify customer segments.
- Evaluate clustering using the Silhouette Score.
- Analyse and visualise the resulting customer groups.

## Dataset

**Dataset:** Mall Customer Segmentation Dataset

The dataset contains 200 customer records with the following attributes:

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Customer's gender |
| `Age` | Customer's age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Spending score assigned to the customer |

The model uses **Annual Income** and **Spending Score** as the primary clustering features.

The dataset is loaded directly from a public CSV source in the notebook.

**Dataset source:** [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation
- **Scikit-learn** — Feature scaling, clustering and evaluation

## Methodology

### 1. Data Loading and Exploration

Load the dataset into a Pandas DataFrame and examine its structure, data types, missing values and statistical summaries.

### 2. Exploratory Data Analysis

Generate visualisations to understand customer age distributions and the relationship between annual income and spending score.

### 3. Feature Selection and Scaling

Select annual income and spending score as the clustering features. Apply `StandardScaler` to standardise the numerical features before clustering.

### 4. Optimal Cluster Selection

Use the Elbow Method to examine the within-cluster sum of squared distances (WCSS) for different values of K and identify a suitable number of clusters.

### 5. K-Means Clustering

Apply the K-Means algorithm to group customers based on similarities in their scaled feature values. The current implementation uses five clusters and `random_state=42` for reproducibility.

### 6. Model Evaluation

Calculate the Silhouette Score to assess how well-separated the resulting clusters are.

### 7. Cluster Analysis

Visualise the identified clusters and summarise their characteristics using customer counts, average ages, annual incomes and spending scores.

## Results and Visualisation

The notebook generates the following outputs:

- **Customer Age Distribution:** A histogram showing the distribution of customer ages.
- **Annual Income vs Spending Score:** A scatter plot exploring the relationship between the two clustering features.
- **Elbow Method:** A graph to help determine a suitable number of clusters.
- **Customer Segmentation Plot:** A scatter plot displaying customers grouped into clusters using different colours.
- **Cluster Summary:** A table showing the number of customers and average feature values for each cluster.
- **Silhouette Score:** A metric used to evaluate cluster separation.

The resulting groups can help illustrate different combinations of income and spending behaviour. Their business interpretations are based on the characteristics of each cluster rather than predefined labels.

## Project Structure

```text
ecommerce-customer-segmentation/
├── AIML_Project.ipynb
├── customer_segmentation.png
├── segmented_customers.csv
├── cluster_summary.csv
├── README.md
└── .gitignore
```

### Output Files

| File | Description |
|---|---|
| `AIML_Project.ipynb` | Notebook containing the code, analysis and visualisations |
| `customer_segmentation.png` | Saved customer segmentation graph |
| `segmented_customers.csv` | Customer records with assigned cluster labels |
| `cluster_summary.csv` | Summary statistics for each cluster |

## Installation and Usage

### Prerequisites

- Python 3
- Jupyter Notebook
- Internet access to retrieve the dataset

### 1. Clone the Repository

```bash
git clone https://github.com/Padmashree06/5TH_Sem_AI_ML_Project.git
cd ecommerce-customer-segmentation
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `AIML_Project.ipynb` in your browser.

### 4. Run the Notebook

Execute the cells sequentially to load the dataset, explore the data, perform clustering, evaluate the model and generate visualisations.

The notebook saves the segmentation graph and CSV results in its working directory.

## Limitations

- The dataset contains only 200 customer records and may not represent all retail customer populations.
- Clustering uses only annual income and spending score.
- The number of clusters must be selected before fitting K-Means.
- Cluster labels have no predefined business meanings and require interpretation.
- The dataset does not contain detailed transaction histories, purchase frequencies or customer retention information.

Although this project focuses on retail customer segmentation, the techniques can also be explored in e-commerce applications using appropriate transaction-level data.

## Future Improvements

- Use real e-commerce transaction data for customer segmentation.
- Incorporate Recency, Frequency and Monetary (RFM) features.
- Compare K-Means with hierarchical clustering and DBSCAN.
- Explore additional clustering evaluation metrics.
- Develop an interactive dashboard for exploring customer segments.
- Evaluate the effectiveness of segment-specific marketing strategies.
---

**Built with Python, Pandas, Scikit-learn and Jupyter Notebook.**