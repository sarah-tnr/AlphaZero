import sys
sys.path.append('..')

from src.KulamiLogic import Board
import numpy as np

class KulamiGame():

    def __init__(self):
        self.board = Board()

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                                that will be the input to your neural network)
        """
        self.board = Board()

        return self.board

    def getBoardSize(self):
        """
        Returns:
            (x,y,z): a tuple of board dimensions
        """
        return np.shape(self.board.np_pieces)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # return number of actions
        s = np.shape(self.board.np_pieces)
        return s[0]*s[1]

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player (row, column)

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        row = action[0]
        column = action[1]
        if self.board.np_pieces[row, column, self.board.idxValidMoves] == 0:
            self.board.add_stone(row, column, player)
            return (self.board, -player)

        return (self.board, player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return self.board.get_valid_moves()

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if self.board.check_game_ended():

            result = self.board.get_winner()
            if result[1] == player:
                return 1
            else:
                return -1

        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        state = board.np_pieces[:, :, self.board.idxFieldState]
        board.np_pieces[:, :, self.board.idxFieldState] = player * state
        return board

    def stringRepresentation(self, board):
        state = board.np_pieces[:, :, self.board.idxFieldState]
        return state.tostring()