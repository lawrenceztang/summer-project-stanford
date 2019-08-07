import csv
import chess
import chess.pgn

file = "dataset_new.pgn"
pgn = open(file)

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
    for i in FEN:
        j = 0
        if i == "/":
            k = 0
            j += 1
        elif i.isdigit():
            k += int(i)
        else:
            list[str_to_int(i)][j][k] = i
            k += 1


dataset = []

for i in range(1000):
    game = chess.pgn.read_game(pgn)
    for node in game.mainline():

        dataset.append({})
        dataset[len(dataset) - 1]["board"] = FEN_to_list(node.board().fen())
        headers = game.headers
        dataset[len(dataset) - 1]["winner"] = [headers["Result"][0], headers["Result"][1]]

