from numpy import loadtxt
from time import time
from torch import Tensor, LongTensor, nn, cuda, no_grad
from torch.nn.functional import relu, log_softmax, nll_loss
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import test2 as t

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 13, 256) # Adjust the input size based on your input dimensions
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, 1, 50, 52) # Reshape input to (batch_size, channels, height, width)
        x = relu(self.conv1(x))
        x = self.pool(x)

        x = relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 12 * 13) # Adjust the size based on your input dimensions
        x = self.dropout(x)

        x = relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output



def prepare_data(filename):
    train_data = loadtxt(filename, delimiter=',')
    X = train_data[:, 1:] 
    y = train_data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tensor_X_train = Tensor(X_train).view(-1, 1, 50, 52) # Reshape input to (batch_size, channels, height, width)
    tensor_y_train = LongTensor(y_train)
    tensor_X_test = Tensor(X_test).view(-1, 1, 50, 52) # Reshape input to (batch_size, channels, height, width)
    tensor_y_test = LongTensor(y_test)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader

def train(model, optimizer, dataloader, device):
    model.train()
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_correct = 0
    with no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            print(pred)
            total_loss += nll_loss(output, target, reduction='sum').item()
            num_correct += pred.eq(target.view_as(pred)).sum().item()

    n = len(dataloader.dataset)
    average_loss = total_loss / n
    accuracy = num_correct / n
    return average_loss, accuracy

def predict_label(filename, model, device):
    # Assuming t.convert_to_mfcc returns data in the correct shape for the model
    X = t.convert_to_mfcc(filename, 130)
    
    # Convert to Tensor and reshape if necessary
    X_tensor = Tensor(X).view(-1, 1, 50, 52)  # Adjust dimensions as needed
    
    # Create a DataLoader for the single test point
   

    # Predict
    model.eval()

    with no_grad():
        X_tensor = X_tensor.to(device)
        output = model(X_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

if __name__ == '__main__':
    start_time = time()

    train_dataloader, test_dataloader = prepare_data('output.csv')

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = NeuralNetwork().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    epoch = 1
    while True:
        train(model, optimizer, train_dataloader, device)
        

        current_time = time()
        if epoch > 10:
            break

        epoch += 1

    
    print(f'Runtime: {current_time - start_time}')
    filename = 'Data/data/SpermWhale/9551900P.wav'
    t.create_test_point(filename)
    predicted_label = predict_label(filename, model, device)
    print(f'Predicted Label: {predicted_label}')










#test_loss, test_accuracy = test(model, test_dataloader, device)

# print(f'Epoch {epoch}:')
# print(f'    Loss: {test_loss}')
# print(f'    Accuracy: {test_accuracy*100}%')
# print()