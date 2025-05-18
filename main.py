import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv("Crop_recommendation.csv")
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
X = df.drop("label", axis=1).values
y = df["label"].values

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# PyTorch Neural Network
class CropNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        return self.output(x)

# Initialize model, loss, and optimizer
input_size = X_train_scaled.shape[1]
num_classes = len(le.classes_)
model = CropNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train neural network
batch_size = 32
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(100):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate neural network
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    accuracy_nn = accuracy_score(y_test_tensor.numpy(), predictions.numpy())

# Traditional ML models
def train_evaluate(model, model_name):
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"{model_name:15} Accuracy: {acc:.4f}")
    return acc

# Train and compare all models
results = {
    "Neural Network": accuracy_nn,
    "Random Forest": train_evaluate(RandomForestClassifier(n_estimators=100), "Random Forest"),
    "XGBoost": train_evaluate(XGBClassifier(), "XGBoost"),
    "SVM": train_evaluate(SVC(kernel='rbf'), "SVM")
}

# Print final comparison
print("\n=== Final Accuracy Comparison ===")
for name, acc in results.items():
    print(f"{name:15} {acc:.4f}")
