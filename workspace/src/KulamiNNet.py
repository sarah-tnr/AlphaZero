from src.KulamiLogic import Board

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

class KulamiNNet(nn.Module):
    "Define Neural Network for Kulami Game."

    def __init__(self, game, args):
        # game params
        # Input for NN is Board wit 6 levels
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        # Output of NN is action
        self.action_size = game.getActionSize()
        self.args = args

        super(KulamiNNet, self).__init__()
        self.conv1 = nn.Conv3d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv3d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm3d(args.num_channels)
        self.bn2 = nn.BatchNorm3d(args.num_channels)
        self.bn3 = nn.BatchNorm3d(args.num_channels)
        self.bn4 = nn.BatchNorm3d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y x board_z
        s = s.view(-1, 1, self.board_x, self.board_y, self.board_z)  # batch_size x 1 x board_x x board_y x board_z
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y x board_z
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y x board_z
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2) x (board_z-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4) x (board_z-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)