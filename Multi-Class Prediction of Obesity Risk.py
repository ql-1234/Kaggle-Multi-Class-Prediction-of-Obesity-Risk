import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 数据预处理函数
def preprocess_data(df, is_train=True):
    # 处理分类特征
    cat_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE','SCC', 'CALC', 'MTRANS']
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 标准化数值特征
    num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    if is_train:
        X = df.drop(['NObeyesdad','id'], axis=1)
        y = le.fit_transform(df['NObeyesdad'])
        return X, y, scaler, le
    else:
        return df.drop('id', axis=1), scaler

# 神经网络模型
class ObesityClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ObesityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / len(val_loader.dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    return best_acc

if __name__ == '__main__':
    torch.manual_seed(42)

    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # 预处理
    X_train, y_train, scaler, label_encoder = preprocess_data(train_df)
    X_test, _ = preprocess_data(test_df, is_train=False)
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
    
    # 转换为Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.values), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 初始化模型
    model = ObesityClassifier(input_size=X_train.shape[1], num_classes=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    best_acc=train_model(model, train_loader, val_loader, criterion, optimizer)
    print("best_acc:",best_acc)

    # 预测测试集
    model.load_state_dict(torch.load('best_model.pth'))
    test_tensor = torch.FloatTensor(X_test.values)
    with torch.no_grad():
        outputs = model(test_tensor)
        _, preds = torch.max(outputs, 1)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'NObeyesdad': label_encoder.inverse_transform(preds.numpy())
    })
    submission.to_csv('submission.csv', index=False)
    