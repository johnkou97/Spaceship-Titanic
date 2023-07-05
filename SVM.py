import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Missing age values are filled with the median age
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# Drop missing values
train_data = train_data.dropna(axis=0)

# Encode categorical variables
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Cabin', 'Destination'])  # One-hot encode categorical columns

# Scale numerical features
scaler = StandardScaler()
train_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(train_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])

# Split the data into features (X) and target variable (y)
X = train_data.drop(['PassengerId', 'Transported', 'Name'], axis=1)
y = train_data['Transported']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# Train the model
model = SVC(random_state=1)
model.fit(X_train, y_train)

# Get predictions
preds = model.predict(X_valid)

# Evaluate the model
print("Accuracy:", accuracy_score(y_valid, preds))

# Make predictions on the test set use the same features as the training set.
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'Cabin', 'Destination'])  # One-hot encode categorical columns
test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.transform(test_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
# Ensure the test data is encoded in the same manner as the training data with the align command
X, X_test = X.align(test_data, join='left', axis=1)
print(X.shape)
print(X_test.shape)

# handle missing values
X_test['Age'] = X_test['Age'].fillna(train_data['Age'].median())
X_test = X_test.fillna(0)

test_preds = model.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                            'Transported': test_preds})
output.to_csv('svm.csv', index=False)



