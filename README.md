# Deploying a Scalable ML Pipeline with FastAPI

![Python](https://img.shields.io/badge/python-3.10-blue)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/amonta53/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

## Project Overview

This project demonstrates how to build, test, and deploy a machine learning inference pipeline using modern MLOps practices.

The system trains a classification model using the Adult Census Income dataset, exposes predictions through a FastAPI REST service, and integrates continuous integration (CI) using GitHub Actions to ensure code quality and reliability.

The primary prediction task is determining whether an individual's income exceeds $50,000 per year based on demographic and employment attributes.

This repository includes:

-   data preprocessing pipeline
-   model training workflow
-   automated testing
-   REST API for model inference
-   CI/CD validation through GitHub Actions

## Dataset
This project uses the Adult Census Income dataset, derived from the 1994 U.S. Census.

Key characteristics:
| Attribute | Value |
|-----------|-------|
| Instances | 32,561 |
| Features| 15 |
| Task| Binary classification |
| Target| salary (>50K or <=50K) |
| Feature types| Numeric + Categorical |

The dataset contains demographic and employment features such as:

    - age 
    - education level 
    - occupation 
    - marital status 
    - hours worked per week 
    - capital gains/losses
  Several categorical variables contain missing values represented as `"?"`.  


## Dataset Exploration
Before building the machine learning pipeline, the dataset was profiled to better understand its structure, feature types, and potential data quality issues.

A custom profiling script (`explore_data`) was created to generate a quick diagnostic overview of the dataset.

This function reports:

    -   dataset shape
    -   column names and data types
    -   sample rows
    -   missing values
    -   placeholder missing values (`?`)
    -   categorical feature cardinality
    -   target class distribution
    -   numeric summary statistics
    

Example usage:

    df  =  pd.read_csv("data/census.csv") 
    profile_data(df, target_col="salary")  
This exploration revealed:

-   a mixture of numeric and categorical features
-   class imbalance between income groups
-   placeholder missing values represented by `"?"`
-   highly skewed financial variables such as `capital-gain` and `capital-loss`
    
These findings informed the preprocessing pipeline used in the model training process. 
The dataset contains leading whitespace in several categorical fields.
These are handled programmatically during loading using skipinitialspace=True rather than modifying 
the raw dataset file.

## Project Structure
Deploying-a-Scalable-ML-Pipeline-with-FastAPI/
├── data/
│   └── census.csv
├── ml/
│   ├── data.py
│   └── model.py
├── scripts/
│   └── explore_data.py
├── model/
│   ├── model.pkl
│   └── encoder.pkl
├── .github/
│   └── workflows/
│       └── python-app.yml
├── main.py
├── train_model.py
├── test_ml.py
└── README.md


## Machine Learning Pipeline

The training pipeline consists of the following steps:
**Data Preprocessing**
The `process_data()` function performs:
-   categorical encoding  
-   feature transformation
-   dataset splitting
-   label binarization

**Model Training**
The model is trained using a supervised classification algorithm.
Steps include:
1.  preprocessing dataset
2.  training the model
3.  generating predictions
4.  evaluating performance metrics
5.  saving the trained model artifacts

## Model Evaluation
Model performance is evaluated using:
-   precision
-   recall
-   F1-score
    
Additionally, the project evaluates model performance across **categorical feature slices** to detect potential bias or performance discrepancies.

Slice metrics are saved to:

    slice_output.txt

## API Service
The trained model is exposed via a FastAPI service.

The API allows external applications to submit demographic information and receive income predictions.

**Endpoints**
Root Endpoint -

    GET /
Returns a simple welcome message.

Prediction Endpoint - 

    POST /predict
Example request:

       </> JSON
        {
    		  "age": 37,
    		  "workclass": "Private",
    		  "education": "Bachelors",
    		  "marital-status": "Married-civ-spouse",
    		  "occupation": "Exec-managerial",
    		  "relationship": "Husband",
    		  "race": "White",
    		  "sex": "Male",
    		  "hours-per-week": 40
    	}
Example response:

    </> JSON 
    {
	    "prediction": ">50K" 
    }

## Continuous Integration
This project uses GitHub Actions to automatically validate code changes.

The CI workflow performs:
-   dependency installation 
-   linting with flake8
-   automated testing with pytest

Workflow configuration is located in:

    .github/workflows/python-app.yml

## Running the Project
**Clone the repository**

    git clone <repo-url> 
    cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
**Create the environment**

    conda env create -f environment.yml
    conda activate fastapi
**Train the model**

    python train_model.py
**Run the API**

    Run the API
API will be available at:

    http://127.0.0.1:8000
Interactive documentation:

    http://127.0.0.1:8000/docs

## Testing
Run unit tests:

     pytest
Lint the code:

    flake8

## Model Card
A model card describing training data, evaluation metrics, and ethical considerations is included in this repository.


## Future Improvements
Potential enhancements include:

    - automated model retraining pipeline
    - improved feature engineering
    - model explainability tools
    - containerized deployment using Docker

## Author
Andrew Montalbano  
Student ID: 012821411
D501 - Machine Learning DevOps – WGU
