# Model Card

## Model Details

This project implements a Random Forest classifier designed to predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes from census data. The model was built in Python using the scikit-learn library as part of a machine learning deployment pipeline.

The model itself is an ensemble of decision trees. In this case, the classifier uses 100 trees, each trained on a random subset of the training data and a random subset of features. This approach helps reduce overfitting and generally produces more stable predictions than relying on a single decision tree. The model uses a fixed random state (42) so that results remain reproducible across different runs.

The trained model and its associated preprocessing components are serialized using `joblib` so they can be loaded consistently during inference.

| File | Description |
|------|-------------|
| model/model.pkl | The trained RandomForestClassifier containing 100 decision trees |
| model/encoder.pkl | The fitted OneHotEncoder used to transform categorical features into numeric format |
| model/lb.pkl | The fitted LabelBinarizer used to convert salary labels between string and binary formats |

Separating the model from the preprocessing artifacts ensures that the same transformations used during training are also applied during prediction.

## Intended Use

This model is primarily intended as an educational example of how to train, package, and deploy a machine learning model using a modern API framework (FastAPI). The goal is not to produce a production-grade income predictor but to demonstrate the full machine learning lifecycle: preprocessing, training, evaluation, serialization, and inference.

The model predicts whether an individual earns more than $50,000 annually using demographic and employment features such as age, education level, occupation, and weekly working hours.

The most appropriate users of this model are:
-   Students learning machine learning engineering and deployment workflows
-   Developers experimenting with model APIs
-   Researchers exploring income classification patterns in census-style data

This model should not be used to make real-world decisions that impact individuals, such as hiring, credit approval, insurance pricing, or housing decisions. It has not been audited for fairness or regulatory compliance, and it relies on historical census data that may not reflect current economic conditions.

Because the dataset originates from U.S. census data, the model should also not be applied to populations outside the United States.

## Training Data
The model was trained using the Census Income dataset (Adult dataset) from the UCI Machine Learning Repository. The dataset was originally extracted from the 1994 U.S. Census database and contains demographic and employment information for individuals along with a binary income classification.

The full dataset contains 32,562 records, each with 14 input features and one target variable representing income classification.

The data was split using an 80/20 train-test split, resulting in approximately:

-   ~26,050 samples used for model training
-   ~6,513 samples used for model evaluation
    
The split was performed using `train_test_split` from scikit-learn with a fixed random state of 42 to ensure consistent reproducibility.

The input features consist of a mixture of continuous and categorical variables.

Continuous features include:
-   Age
-   Final weight (fnlwgt)
-   Education years
-   Capital gains
-   Capital losses
-   Hours worked per week
    

Categorical features include:
-   Workclass
-   Education level
-   Marital status
-   Occupation
-   Relationship status
-   Race
-   Sex
-   Native country
    
The target variable represents annual income and has two possible values:
-   <=50K — individuals earning $50,000 or less
-   >50K — individuals earning more than $50,000
    
The dataset is class imbalanced, with roughly:
-   ~75% in the <=50K class
-   ~25% in the >50K class
    
### Preprocessing
Categorical features are transformed using One-Hot Encoding. The encoder is configured with `handle_unknown="ignore"` so that unseen categories encountered during inference do not cause errors.

The target label is converted into a binary format using a LabelBinarizer, mapping:
-   <=50K → 0
-   > 50K → 1
    
Continuous features are not scaled because Random Forest models are generally insensitive to feature scaling.

## Evaluation Data
Model evaluation was performed on the 20% holdout test set (6,513 samples) that was excluded from the training process.

The evaluation data was transformed using the same encoder and label binarizer that were fitted on the training data. This ensures that the test evaluation accurately reflects real inference conditions and prevents any information leakage between training and evaluation phases.

The class distribution of the test set mirrors the original dataset, maintaining the approximate 75/25 class imbalance. Maintaining this distribution provides a realistic view of how the model performs on unseen data drawn from the same population.

## Metrics
Because the dataset is imbalanced, model evaluation focuses on Precision, Recall, and F1-Score rather than raw accuracy.

Accuracy alone would be misleading in this scenario. A trivial model that always predicts the majority class (<=50K) would achieve around 75% accuracy while completely failing to identify higher earners.

The evaluation metrics capture more meaningful aspects of performance:

Precision measures how often the model is correct when it predicts the >50K class.

Recall measures how well the model identifies all individuals who actually earn >50K.

F1-Score combines both metrics using their harmonic mean, providing a balanced measure that penalizes both false positives and false negatives.

### Overall Model Performance
Based on evaluation on the holdout test set, the model achieves the following overall performance metrics: 

| Metric    | Score   | 
|--------   |---------|
| Precision | 0.74    | 
| Recall    | 0.64    | 
| F1-Score  | 0.68    |

The model demonstrates moderate predictive performance, with relatively strong precision but weaker recall, indicating it is better at confirming high earners than identifying all of them.

### Performance by Workclass
The model shows varying performance across different employment types. Federal government employees show the best performance with an F1-score of 0.7914, while self-employed individuals not incorporated show lower performance at 0.5789.

| Workclass        | Precision | Recall | F1-Score | Count |
|-----------       |-----------|--------|----------|-------|
| Without-pay      | 1.0000    | 1.0000 | 1.0000   | 4     |  
| Federal-gov      | 0.7971    | 0.7857 | 0.7914   | 191   | 
| Self-emp-inc     | 0.7807    | 0.7542 | 0.7672   | 212   | 
| Local-gov        | 0.7576    | 0.6818 | 0.7177   | 387   |
| State-gov        | 0.7424    | 0.6712 | 0.7050   | 254   |
| Private          | 0.7376    | 0.6404 | 0.6856   | 4578  |
| Self-emp-not-inc | 0.7064    | 0.4904 | 0.5789   | 498   | 
| ? (Unknown)      | 0.6538    | 0.4048 | 0.5000   | 389   |

### Performance by Education Level
Education level significantly impacts model performance. The model performs best for individuals with advanced degrees and struggles with lower education levels.
| Education   | Precision | Recall | F1-Score | Count |
| ----------- | --------- | ------ | -------- | ----- |
| 1st-4th     | 1.0000    | 1.0000 | 1.0000   | 23    |
| Preschool   | 1.0000    | 1.0000 | 1.0000   | 10    |
| Prof-school | 0.8182    | 0.9643 | 0.8852   | 116   |
| Doctorate   | 0.8644    | 0.8947 | 0.8793   | 77    |
| Masters     | 0.8271    | 0.8551 | 0.8409   | 369   |
| Bachelors   | 0.7523    | 0.7289 | 0.7404   | 1053  |
| Some-college| 0.6857    | 0.5199 | 0.5914   | 1485  |
| HS-grad     | 0.6594    | 0.4377 | 0.5261   | 2085  |
| 11th        | 1.0000    | 0.2727 | 0.4286   | 225   |
| 10th        | 0.4000    | 0.1667 | 0.2353   | 183   |
| 7th-8th     | 0.0000    | 0.0000 | 0.0000   | 141   |

### Performance by Sex
The model shows a notable performance gap between male and female individuals. Male individuals have an F1-score of 0.6997 compared to 0.6015 for female individuals, representing a difference of approximately 9.8 percentage points.
| Sex    | Precision | Recall | F1-Score | Count |
| ----   | --------- | ------ | -------- | ----- |
| Male   | 0.7445    | 0.6599 | 0.6997   | 4387  |
| Female | 0.7229    | 0.5150	| 0.6015	 | 2126  |


### Performance by Race
Performance varies across racial groups. Asian-Pac-Islander individuals show the highest F1-score at 0.7458, while Amer-Indian-Eskimo individuals show the lowest at 0.5556 among groups with sufficient sample size.

| Race               | Precision | Recall | F1-Score | Count |
| ------------------ | --------- | ------ | -------- | ----- |
| Other              | 1.0000    | 0.6667 | 0.8000   | 55    |
| Asian-Pac-Islander | 0.7857    | 0.7097 | 0.7458   | 193   |
| White              | 0.7404    | 0.6373 | 0.6850   | 5595  |
| Amer-Indian-Eskimo | 0.6250    | 0.5000 | 0.5556   | 71    |
| Black              | 0.7273    | 0.6154 | 0.6667   | 599   |


### Performance by Occupation
Executive and managerial positions show the best performance with an F1-score of 0.7736, while farming and fishing occupations show lower performance at 0.3077.
| Occupation        | Precision | Recall | F1-Score | Count |
| ---------------   | --------- | ------ | -------- | ----- |
| Armed-Forces      | 1.0000    | 1.0000 | 1.0000   | 3     |
| Priv-house-serv   | 1.0000    | 1.0000 | 1.0000   | 26    |
| Prof-specialty    | 0.7880    | 0.7679 | 0.7778   | 828   |
| Exec-managerial   | 0.7952    | 0.7531 | 0.7736   | 838   | 
| Tech-support      | 0.7143    | 0.6863 | 0.7000   | 189   | 
| Farming-fishing   | 0.5455    | 0.2143 | 0.3077   | 193   |
| Other-service     | 1.0000    | 0.1923 | 0.3226   | 667   |
| Handlers-cleaners | 0.5714    | 0.3333 | 0.4211   | 273   |


## Ethical Considerations
Several patterns in the evaluation results highlight potential fairness concerns.

First, the model shows a noticeable performance gap between male and female individuals. Differences like this may indicate that the model is capturing historical biases embedded in the training data rather than true causal relationships.

Second, some demographic groups have relatively small sample sizes, which can produce unstable model behavior and inconsistent predictions.

Another important factor is the age of the training data. The dataset reflects socioeconomic conditions from 1994. Labor markets, education access, and income distributions have changed significantly since then. Predictions based on this dataset should therefore be interpreted cautiously.

The model also includes protected attributes such as race and sex as input features. While including these attributes may improve predictive accuracy, their use in income prediction tasks raises ethical and legal concerns. In real-world deployments, these variables would likely need to be removed or carefully evaluated.

## Caveats and Recommendations
Several limitations should be considered when interpreting results from this model.

The dataset is more than three decades old and may not reflect current labor markets or income dynamics. As a result, model predictions should be treated as illustrative rather than authoritative.

The overall F1-score indicates that the model misclassifies a meaningful portion of observations. In particular, recall for the >50K class is relatively modest, meaning the model fails to identify a significant number of high earners.

Performance also varies across demographic subgroups. Some groups show significantly weaker results, which suggests that the model may not generalize equally well across the entire population.

Finally, the Random Forest was trained using mostly default parameters. Additional performance gains might be possible through systematic hyperparameter tuning.

Future improvements could include:
-   Hyperparameter optimization using grid search or randomized search
-   Feature importance analysis to understand the most influential predictors
-   Bias mitigation techniques to address subgroup disparities
-   Training on more recent census data
-   Model explainability tools such as SHAP or LIME
    
Before deploying any model in a real-world setting, careful bias analysis, updated training data, and human oversight would be essential.