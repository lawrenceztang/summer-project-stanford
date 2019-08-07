import run
import torch
import pickle
import chess
import chess.svg
from draw import DrawChessPosition

save_path = "model.pth"
test_data_path = "test_data"

net = run.Net()
net.load_state_dict(torch.load(save_path))

with open(test_data_path, "rb") as fp:
    test_data = pickle.load(fp)

for i in range(100):
    renderer = DrawChessPosition()
    fen = test_data[2][i]
    b = renderer.draw(fen[0:fen.find(" ")])
    b.save("boards/" + str(int(round(net(torch.FloatTensor([test_data[0][i]])).data.numpy()[0][0]))) + "/" + str(i) + ".png")