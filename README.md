# Kaggle Titanic
The goal of this repository is to store the different models I tried so far for the Kaggle Titanic competition.

I tried several models and several approaches, including Feature selection and Feature engineering.

I was inspired by this kernel: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

Currently, my best score is 0.78469, obtained with a Logistic Regression.

## Experiments
In the folder scripts/experiment, I reused the code from my starter-code-for-ML repository (https://github.com/gcoter/starter-code-for-ML) to test different models.

Here are the results:

```
                                        Hyperparameters         Training Accuracy (%)       Validation Accuracy (%)         Training Duration (s)
SKLearnLogisticRegression          C=1.0,max_iter=10000                         79.43                          76.4                             0
SKLearnLogisticRegression           C=10,max_iter=10000                          79.8                         75.28                             0
SKLearnLogisticRegression           C=20,max_iter=10000                          79.8                         75.28                             0
SKLearnLogisticRegression           C=30,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=40,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=50,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=60,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=70,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=80,max_iter=10000                         80.55                          80.9                             0
SKLearnLogisticRegression           C=90,max_iter=10000                         80.55                          80.9                             0
           SKRandomForest               n_estimators=10                         86.91                         75.28                             0
           SKRandomForest              n_estimators=100                         87.41                         74.16                             0
           SKRandomForest              n_estimators=200                         87.41                         75.28                             0
           SKRandomForest              n_estimators=300                         87.41                         75.28                             0
           SKRandomForest              n_estimators=400                         87.41                         73.03                             0
           SKRandomForest              n_estimators=500                         87.41                         75.28                             1
           SKRandomForest              n_estimators=600                         87.41                         73.03                             1
           SKRandomForest              n_estimators=700                         87.41                         73.03                             1
           SKRandomForest              n_estimators=800                         87.41                         75.28                             1
           SKRandomForest              n_estimators=900                         87.41                         75.28                             2
             SKNaiveBayes                                                        79.3                         77.53                             0
```

The best model is then selected to generate the submission file (SKLearnLogisticRegression (C=30,max_iter=10000)).

## Requirements
Pandas
scikit-learn

## References
Competition : https://www.kaggle.com/c/digit-recognizer