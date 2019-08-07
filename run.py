import csv
import chess
from tqdm import trange
from random import shuffle
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(12, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 4 * 16, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12, 700)  # 6*6 from image dimension
        self.fc2 = nn.Linear(700, 500)
        self.fc3 = nn.Linear(500, 400)
        self.fc4 = nn.Linear(400, 300)
        self.fc5 = nn.Linear(300, 2)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(x)
        return x

layer_pawn_w = 0
layer_knight_w = 1
layer_bishop_w = 2
layer_rook_w = 3
layer_queen_w = 4
layer_king_w = 5
layer_pawn_b = 6
layer_knight_b = 7
layer_bishop_b = 8
layer_rook_b = 9
layer_queen_b = 10
layer_king_b = 11

board_size = 8

def str_to_int(str):
    if str == "P":
        return layer_pawn_w
    if str == "p":
        return layer_pawn_b
    if str == "N":
        return layer_knight_w
    if str == "n":
        return layer_knight_b
    if str == "B":
        return layer_bishop_w
    if str == "b":
        return layer_bishop_b
    if str == "R":
        return layer_rook_w
    if str == "r":
        return layer_rook_b
    if str == "Q":
        return layer_queen_w
    if str == "q":
        return layer_queen_b
    if str == "K":
        return layer_king_w
    if str == "k":
        return layer_king_b

def FEN_to_list(FEN):
    list = [[[0 for k in range(board_size)] for j in range(board_size)] for i in range(12)]
    k = 0
    j = 0
    for i in FEN:
        if i == " ":
            return list
        if i == "/":
            k = 0
            j += 1
        elif i.isdigit():
            k += int(i)
        else:
            list[str_to_int(i)][j][k] = 1
            k += 1
    return list

test_data_path = "test_data"
save_path = "model.pth"
csv_file = "games.csv"
csv_reader = csv.DictReader(open(csv_file), delimiter=",")
full_file = []
for row in csv_reader:
    full_file.append(row)

def main():

    dataset = [[], [], []]
    test_dataset = [[], [], []]

    for i in trange(0, 3000):
        board = chess.Board()
        moves = full_file[i]["moves"]

        for move in moves.split():

            board.push_san(move)

            dataset[0].append([])
            dataset[1].append([])
            dataset[0][len(dataset[0]) - 1] = FEN_to_list(board.fen())
            dataset[2].append(board.fen())
            dataset[1][len(dataset[1]) - 1] = [0, 0]
            dataset[1][len(dataset[1]) - 1][0] = .5 if full_file[i]["winner"] == "draw" else 0
            dataset[1][len(dataset[1]) - 1][0] = 1 if full_file[i]["winner"] == "white" else 0
            dataset[1][len(dataset[1]) - 1][1] = .5 if full_file[i]["winner"] == "draw" else 0
            dataset[1][len(dataset[1]) - 1][1] = 0 if full_file[i]["winner"] == "white" else 1

    for i in trange(18000, 18500):
        board = chess.Board()
        moves = full_file[i]["moves"]

        for move in moves.split():

            board.push_san(move)

            test_dataset[0].append([])
            test_dataset[1].append([])
            test_dataset[0][len(test_dataset[0]) - 1] = FEN_to_list(board.fen())
            test_dataset[2].append(board.fen())
            test_dataset[1][len(test_dataset[1]) - 1] = [0, 0]
            test_dataset[1][len(test_dataset[1]) - 1][0] = .5 if full_file[i]["winner"] == "draw" else 0
            test_dataset[1][len(test_dataset[1]) - 1][0] = 1 if full_file[i]["winner"] == "white" else 0
            test_dataset[1][len(test_dataset[1]) - 1][1] = .5 if full_file[i]["winner"] == "draw" else 0
            test_dataset[1][len(test_dataset[1]) - 1][1] = 0 if full_file[i]["winner"] == "white" else 1

    r = torch.randperm(len(dataset[0])).numpy()
    dataset[0] = [dataset[0][r[i]] for i in range(len(dataset[0]))]
    dataset[1] = [dataset[1][r[i]] for i in range(len(dataset[0]))]
    dataset[2] = [dataset[2][r[i]] for i in range(len(dataset[0]))]

    train_x = torch.FloatTensor(dataset[0])
    train_y = torch.FloatTensor(dataset[1])


    with open(test_data_path,  "wb") as fp:
        pickle.dump(test_dataset, fp)

    test_x = torch.FloatTensor(test_dataset[0])
    test_y = torch.FloatTensor(test_dataset[1])

    net = Net()
    net.load_state_dict(torch.load(save_path))

    batch_size = 10
    criterion = nn.MSELoss()
    learning_rate = 0.02
    epochs = 3

    for k in trange(epochs):

        with torch.no_grad():
            out_test = net(test_x)
            test_loss = criterion(out_test, test_y)
            print(test_loss.item())

        for i in range(len(dataset[0]) // batch_size):

            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            output = net(train_x[i * batch_size:(i + 1) * batch_size])

            loss = criterion(output, train_y[i * batch_size:(i + 1) * batch_size])
            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), save_path)

    with torch.no_grad():
        test_right = 0
        for j in range(test_y.size()[0]):
            guess = int(round(net(test_x[j:j+1]).numpy()[0][0]))
            truth = test_y[j][0].numpy()
            if abs(guess - truth) < .001:
                test_right += 1
        print(test_right / test_y.size()[0])

if __name__ == "__main__":
    main()
