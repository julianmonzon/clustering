# Clustering Analysis with K-Means and K-Medoids

These scripts perform clustering analysis on a datasets, either by loading an existing dataset or generating synthetic data. The clustering is done using **K-Means** and **K-Medoids** algorithms, and visualizations are provided to interpret the results.

---

## Workflow

The script follows these steps:

1. **Load or Create Dataset**  
   Load a predefined dataset (e.g., Iris) or generate synthetic data.  

2. **Data Normalization**  
   Standardize the dataset to ensure consistency and improve clustering performance.  

3. **Create DataFrame**  
   Organize the dataset into a Pandas DataFrame for easier handling and visualization.  

4. **Visualize Original Data**  
   Plot the initial dataset to understand its structure.  

5. **Apply K-Means Algorithm**  
   Cluster the dataset using K-Means and compute the resulting labels.  

6. **Visualize K-Means Results**  
   Display the clustering results from the K-Means algorithm.  

7. **Apply K-Medoids Algorithm**  
   Cluster the dataset using K-Medoids for an alternative approach.  

8. **Visualize K-Medoids Results**  
   Display the clustering results from the K-Medoids algorithm.  

---

## Required Libraries

The script requires the following Python libraries:  

- **numpy**: For numerical computations.  
- **pandas**: For data handling and manipulation.  
- **scikit-learn**: For K-Means and preprocessing utilities.  
- **sklearn-extra**: For K-Medoids clustering.  
- **matplotlib**: For data visualization.  
- **seaborn**: For enhanced data plotting.

---

## Instructions

1. **Install Dependencies**  
   Install the required libraries using the `requirements.txt` file. Run the following command:  
   ```bash
   pip install -r requirements.txt

2. **Run the Script**

    Execute the script to load the dataset, perform clustering, and view visualizations of the clustering results:

    ```bash
    python example.py

