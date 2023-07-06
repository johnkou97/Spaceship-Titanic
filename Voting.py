import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
preds_rf = pd.read_csv('submissions/random_forest_2000.csv')['Transported'].astype(int)
preds_gbm = pd.read_csv('submissions/gbm.csv')['Transported'].astype(int)
preds_ada = pd.read_csv('submissions/ada.csv')['Transported'].astype(int)
preds_bagging = pd.read_csv('submissions/bagging.csv')['Transported'].astype(int)
preds_stacking = pd.read_csv('submissions/stacking.csv')['Transported'].astype(int)
preds_neural = pd.read_csv('submissions/neural.csv')['Transported'].astype(int)
preds_svm = pd.read_csv('submissions/svm.csv')['Transported'].astype(int)


# Majority voting
preds = preds_rf + preds_gbm + preds_ada + preds_bagging + preds_stacking + preds_neural + preds_svm
preds = preds / 7
preds = preds.round().astype(int)
preds = preds.astype(bool)
# compare with gbm how many are the same
print("Accuracy:", accuracy_score(preds_gbm, preds))


# Save test predictions to file
output = pd.DataFrame({'PassengerId': pd.read_csv('data/test.csv').PassengerId,
                            'Transported': preds})
output.to_csv('voting.csv', index=False)

# create one csv file with all the predictions
output = pd.DataFrame({'PassengerId': pd.read_csv('data/test.csv').PassengerId,
                        'forest': preds_rf,
                        'gbm': preds_gbm,
                        'ada': preds_ada,
                        'bagging': preds_bagging,
                        'stacking': preds_stacking,
                        'neural': preds_neural,
                        'svm': preds_svm,
                        'voting': preds})
output.to_csv('all.csv', index=False) 

