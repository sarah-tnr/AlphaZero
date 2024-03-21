# from src.KulamiLogic import Board

import os
os.environ["KERAS_BACKEND"] = "torch"
from KulamiGame import KulamiGame
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchvision import models
from torchsummary import summary
from dotdict import dotdict
import keras
from keras.models import *
from keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, Dropout, Softmax, Concatenate, Add
from keras.layers import Conv2D
from keras.optimizers import Adam


class KulamiNNet(nn.Module):
    "Define Neural Network for Kulami Game."

    def __init__(self, game, args):
        # game params
        # Input for NN is Board wit 6 levels
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        print(self.board_x, self.board_y, self.board_z)
        # Output of NN is action
        self.action_size = game.getActionSize()
        self.args = args

        super(KulamiNNet, self).__init__()
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.board_z))
        self.conv1 = Conv2D(args.num_channels, activation="relu", kernel_size=3, strides=1, padding="same")(self.input_boards)
        self.bn1 = BatchNormalization()(self.conv1)
        self.conv2 = Conv2D(args.num_channels, activation="relu", kernel_size=3, strides=1, padding="same")(self.bn1)
        self.bn2 = BatchNormalization()(self.conv2)
        self.conv3 = Conv2D(args.num_channels, activation="relu", kernel_size=3, strides=1, padding="valid")(self.bn2)
        self.bn3 = BatchNormalization()(self.conv3)
        self.conv4 = Conv2D(args.num_channels, activation="relu", kernel_size=3, strides=1, padding="valid")(self.bn3)
        self.bn4 = BatchNormalization()(self.conv4)

        self.flatten = Flatten()(self.bn4)

        self.fc1 = Dense(1024, activation="relu")(self.flatten)
        self.fc_bn1 = BatchNormalization()(self.fc1)
        self.fc_dropout1 = Dropout(args.dropout)(self.fc_bn1)

        self.fc2 = Dense(512, activation="relu")(self.fc_dropout1)
        self.fc_bn2 = BatchNormalization()(self.fc2)
        self.fc_dropout2 = Dropout(args.dropout)(self.fc_bn2)

        self.fc3 = Dense(self.action_size, activation="softmax", name="policy")(self.fc_dropout2)
        self.fc4 = Dense(1, activation="tanh", name="value")(self.fc_dropout2)

        self.model = Model(inputs=self.input_boards, outputs=[self.fc3, self.fc4])
        self.model.summary()

    # (batch size, 224, 224, 3)
    # (batch size, 3, 224, 244)
    # np.transpose(x, (0, 3, 1, 2))


if __name__ == '__main__':
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 256,
    })

    game = KulamiGame()
    network = KulamiNNet(args=args, game=game)
    # summary(network, (6, 8, 8), batch_size=1)
    # test_tensor = torch.zeros((1, 6, 8, 8))
    # output = network(test_tensor)
