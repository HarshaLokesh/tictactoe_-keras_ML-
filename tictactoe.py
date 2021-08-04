# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('C:/Users/Harsha/Desktop/CV/projects/ML/tic-tac-toe-endgame.csv')

# print(df.head())
# print(df.describe())

# X = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9']]
# y = df['V10']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

import numpy as np 
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.backend import reshape
from keras.utils.np_utils import to_categorical
import random


def initBoard():
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    return board

def getMoves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves

def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            mark = ' '
            if board[i][j] == 1:
                mark = 'X'
            elif board[i][j] == 2:
                mark = 'O'
            if (j == len(board[i]) - 1):
                print(mark)
            else:
                print(str(mark) + "|", end='')
        if (i < len(board) - 1):
            print("-----")


def getWinner(board):
    candidate = 0
    won = 0
    
    # Check rows
    for i in range(len(board)):
        candidate = 0
        for j in range(len(board[i])):
            
            # Make sure there are no gaps
            if board[i][j] == 0:
                break
            
            # Identify the front-runner
            if candidate == 0:
                candidate = board[i][j]
            
            # Determine whether the front-runner has all the slots
            if candidate != board[i][j]:
                break
            elif j == len(board[i]) - 1:
                won = candidate
    
    if won > 0:
        return won
    
    # Check columns
    for j in range(len(board[0])):
        candidate = 0
        for i in range(len(board)):
            
            # Make sure there are no gaps
            if board[i][j] == 0:
                break
            
            # Identify the front-runner
            if candidate == 0:
                candidate = board[i][j]
            
            # Determine whether the front-runner has all the slots
            if candidate != board[i][j]:
                break
            elif i == len(board) - 1:
                won = candidate
    
    if won > 0:
        return won
    
    # Check diagonals
    candidate = 0
    for i in range(len(board)):
        if board[i][i] == 0:
            break
        if candidate == 0:
            candidate = board[i][i]
        if candidate != board[i][i]:
            break
        elif i == len(board) - 1:
            won = candidate
    
    if won > 0:
        return won
    
    candidate = 0
    for i in range(len(board)):
        if board[2 - i][2 - i] == 0:
            break
        if candidate == 0:
            candidate = board[2 - i][2 - i]
        if candidate != board[2 - i][2 - i]:
            break
        elif i == len(board) - 1:
            won = candidate
    
    if won > 0:
        return won
    
    # Still no winner?
    if (len(getMoves(board)) == 0):
        # It's a draw
        return 0
    else:
        # Still more moves to make
        return -1

def getModel():
    numCells = 9
    outcomes = 3
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(9, )))
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def movesToBoard(moves):
    board = initBoard()
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board

def gamesToWinLossData(games):
    X = []
    y = []
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    X = np.array(X).reshape((-1, 9))
    y = to_categorical(y)
    
    # Return an appropriate train/test split
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])

def bestMove(board, model, player, rnd=0):
    scores = []
    moves = getMoves(board)
    
    # Make predictions for each possible move
    for i in range(len(moves)):
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = player
        prediction = model.predict(future.reshape((-1, 9)))[0]
        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # Choose the best move with a random factor
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    # Choose a move completely at random
    return moves[random.randint(0, len(moves) - 1)]

def simulateGame(p1=None, p2=None, rnd=0):
    history = []
    board = initBoard()
    playerToMove = 1
    
    while getWinner(board) == -1:
        
        # Chose a move (random or use a player model if provided)
        move = None
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
        else:
            moves = getMoves(board)
            move = moves[random.randint(0, len(moves) - 1)]
        
        # Make the move
        board[move[0]][move[1]] = playerToMove
        
        # Add the move to the history
        history.append((playerToMove, move))
        
        # Switch the active player
        playerToMove = 1 if playerToMove == 2 else 2
        
    return history

games = [simulateGame() for _ in range(10000)]

model = getModel()
X_train, X_test, y_train, y_test = gamesToWinLossData(games)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100)

board = initBoard()



for i in range(9):
    move = bestMove(board, model, 1)
    board[move[0]][move[1]] = 1
    printBoard(board)


    if getWinner(board) > 0:
        break
    elif getWinner(board) == 0:
        break
    
    print("enter the row:")
    r = int(input())
    print("enter the col:")
    c = int(input())
    

    board[r][c] = 2
    printBoard(board)

    
