import torch.nn as nn
import torch.nn.functional as F


class GestureClassifier(nn.Module):
  def __init__(self, input_size, num_classes):
    super(GestureClassifier, self).__init__()

    self.fc1 = nn.Linear(input_size, 512)
    self.bn1 = nn.BatchNorm1d(512)

    self.fc2 = nn.Linear(512, 256)
    self.bn2 = nn.BatchNorm1d(256)

    # self.fc3 = nn.Linear(512, 256)
    # self.bn3 = nn.BatchNorm1d(256)
    #
    self.fc3 = nn.Linear(256, 128)
    self.bn3 = nn.BatchNorm1d(128)

    self.fc4 = nn.Linear(128, num_classes)

    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = F.relu(self.bn1(self.fc1(x)))
    x = self.dropout(x)

    x = F.relu(self.bn2(self.fc2(x)))
    x = self.dropout(x)

    x = F.relu(self.bn3(self.fc3(x)))
    x = self.dropout(x)
    #
    # x = F.relu(self.bn4(self.fc4(x)))
    # x = self.dropout(x)

    x = self.fc4(x)
    return x
