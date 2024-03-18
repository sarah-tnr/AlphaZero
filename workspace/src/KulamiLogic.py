import sys
sys.path.append('..')

import numpy as np


IDX_SIZEPLATE = 0  # Größe der Holzplatte
IDX_NUMPLATE = 1  # Nummer der Holzplatte
IDX_NUMFIELDONPLATE = 2  # Nummer des Feldes auf der Holzplatte

IDX_FIELDSTATE = 3  # Zustand des Feldes (frei, Spieler 1, Spieler 2)
IDX_CNTMOVE = 4  # Spielzug (Nummer des Spielzugs, wo das Feld besetzt wurde)
IDX_VALIDMOVE = 5  # Zustand, ob Feld besetzt werden darf)

PLAYERS = [-1, 1]

class Board:
    """
    Basic Kulami Board.
    """

    def __init__(self, height=None, width=None, np_pieces=None):
        """Set up initial board configuration."""

        DEFAULT_HEIGHT = 8
        DEFAULT_WIDTH = 8

        self.idxSizePlate = 0
        self.idxNumPlate = 1
        self.idxNumFieldOnPlate = 2
        self.idxFieldState = 3
        self.idxCntMove = 4
        self.idxValidMoves = 5
        self.players = [-1,1]

        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH

        self.cnt_moves = 0

        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width], dtype=np.dtype('(6,)i4'))

        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

        "Set up fields on Board"
        "4x 2x2 fields"
        self.np_pieces[0, 0, :3] = (4, 1, 1)
        self.np_pieces[0, 1, :3] = (4, 1, 2)
        self.np_pieces[1, 0, :3] = (4, 1, 3)
        self.np_pieces[1, 1, :3] = (4, 1, 4)

        self.np_pieces[2, 0, :3] = (4, 2, 1)
        self.np_pieces[2, 1, :3] = (4, 2, 2)
        self.np_pieces[3, 0, :3] = (4, 2, 3)
        self.np_pieces[3, 1, :3] = (4, 2, 4)

        self.np_pieces[1, 3, :3] = (4, 3, 1)
        self.np_pieces[1, 4, :3] = (4, 3, 2)
        self.np_pieces[2, 3, :3] = (4, 3, 3)
        self.np_pieces[2, 4, :3] = (4, 3, 4)

        self.np_pieces[4, 5, :3] = (4, 4, 1)
        self.np_pieces[4, 6, :3] = (4, 4, 2)
        self.np_pieces[5, 5, :3] = (4, 4, 3)
        self.np_pieces[5, 6, :3] = (4, 4, 4)

        self.np_pieces[6, 6, :3] = (4, 5, 1)
        self.np_pieces[6, 7, :3] = (4, 5, 2)
        self.np_pieces[7, 6, :3] = (4, 5, 3)
        self.np_pieces[7, 7, :3] = (4, 5, 4)

        "4x 3x3 fields"
        self.np_pieces[5, 1, :3] = (6, 1, 1)
        self.np_pieces[5, 2, :3] = (6, 1, 2)
        self.np_pieces[6, 1, :3] = (6, 1, 3)
        self.np_pieces[6, 2, :3] = (6, 1, 4)
        self.np_pieces[7, 1, :3] = (6, 1, 5)
        self.np_pieces[7, 2, :3] = (6, 1, 6)

        self.np_pieces[3, 2, :3] = (6, 2, 1)
        self.np_pieces[3, 3, :3] = (6, 2, 2)
        self.np_pieces[3, 4, :3] = (6, 2, 3)
        self.np_pieces[4, 2, :3] = (6, 2, 4)
        self.np_pieces[4, 3, :3] = (6, 2, 5)
        self.np_pieces[4, 4, :3] = (6, 2, 6)

        self.np_pieces[6, 3, :3] = (6, 3, 1)
        self.np_pieces[6, 4, :3] = (6, 3, 2)
        self.np_pieces[6, 5, :3] = (6, 3, 3)
        self.np_pieces[7, 3, :3] = (6, 3, 4)
        self.np_pieces[7, 4, :3] = (6, 3, 5)
        self.np_pieces[7, 5, :3] = (6, 3, 6)

        self.np_pieces[0, 6, :3] = (6, 4, 1)
        self.np_pieces[1, 7, :3] = (6, 4, 2)
        self.np_pieces[2, 6, :3] = (6, 4, 3)
        self.np_pieces[0, 7, :3] = (6, 4, 4)
        self.np_pieces[1, 6, :3] = (6, 4, 5)
        self.np_pieces[2, 7, :3] = (6, 4, 6)

        "4x 3x1 fields"
        self.np_pieces[0, 2, :3] = (3, 1, 1)
        self.np_pieces[1, 2, :3] = (3, 1, 2)
        self.np_pieces[2, 2, :3] = (3, 1, 3)

        self.np_pieces[0, 3, :3] = (3, 2, 1)
        self.np_pieces[0, 4, :3] = (3, 2, 2)
        self.np_pieces[0, 5, :3] = (3, 2, 3)

        self.np_pieces[5, 0, :3] = (3, 3, 1)
        self.np_pieces[6, 0, :3] = (3, 3, 2)
        self.np_pieces[7, 0, :3] = (3, 3, 3)

        self.np_pieces[3, 5, :3] = (3, 4, 1)
        self.np_pieces[3, 6, :3] = (3, 4, 2)
        self.np_pieces[3, 7, :3] = (3, 4, 3)

        "4x 2x1 fields"
        self.np_pieces[4, 0, :3] = (2, 1, 1)
        self.np_pieces[4, 1, :3] = (2, 1, 2)

        self.np_pieces[5, 3, :3] = (2, 2, 1)
        self.np_pieces[5, 4, :3] = (2, 2, 2)

        self.np_pieces[1, 5, :3] = (2, 3, 1)
        self.np_pieces[2, 5, :3] = (2, 3, 2)

        self.np_pieces[4, 7, :3] = (2, 4, 1)
        self.np_pieces[5, 7, :3] = (2, 4, 2)

    def choose_player(self):
        """Get player for the next move."""

        if not np.any(self.np_pieces[:, :, IDX_FIELDSTATE]):
            "Start with a random player"
            return np.random.choice(PLAYERS)

        else:
            "Switch players"
            idx_lastplayer = np.where(self.np_pieces[:, :, IDX_CNTMOVE] == self.cnt_moves)
            if self.np_pieces[idx_lastplayer[0], idx_lastplayer[1], IDX_FIELDSTATE] == PLAYERS[0]:
                return PLAYERS[1]
            elif self.np_pieces[idx_lastplayer[0], idx_lastplayer[1], IDX_FIELDSTATE] == PLAYERS[1]:
                return PLAYERS[0]
            else:
                raise ValueError("Next player could not be identified.")

    def get_valid_moves(self):
        """Any zero value is a valid move."""

        return self.np_pieces[:, :, IDX_VALIDMOVE]

    def add_stone(self, row, column, player):
        """Add a new stone on board and update dependencies."""

        "Check if move is valid."
        if self.np_pieces[row, column, IDX_VALIDMOVE] == 0:

            "Place player on valid field"
            self.np_pieces[row, column, IDX_FIELDSTATE] = player

            "Write number of move to field"
            self.cnt_moves = self.cnt_moves + 1
            self.np_pieces[row, column, IDX_CNTMOVE] = self.cnt_moves

            "Update next valid moves"
            self.np_pieces[:, :, IDX_VALIDMOVE] = 1 # default, all moves are invalid

            "1. Rule - next move must be in same row or column of last move"
            self.np_pieces[row, :, IDX_VALIDMOVE] = 0  # place in same row is valid move
            self.np_pieces[:, column, IDX_VALIDMOVE] = 0  # place in same column is valid move

            "2. Rule - next move must be in another field than the two sub fields before"
            # Get indices of moves in current field
            idx_invalidfield1 = (self.np_pieces[:, :, 0] == self.np_pieces[row, column, 0]) & (
                    self.np_pieces[:, :, 1] == self.np_pieces[row, column, 1])
            idx_invalidfield = idx_invalidfield1
            if self.cnt_moves > 1:
                lastmove = np.where(self.np_pieces[:, :, IDX_CNTMOVE] == self.cnt_moves - 1)
                idx_invalidfield2 = (self.np_pieces[:, :, 0] == self.np_pieces[lastmove[0], lastmove[1], 0]) & (
                        self.np_pieces[:, :, 1] == self.np_pieces[lastmove[0], lastmove[1], 1])
                idx_invalidfield = idx_invalidfield | idx_invalidfield2

            "Fields that are not free needs to be excluded"
            idx_invalidfield3 = self.np_pieces[:, :, IDX_FIELDSTATE] != 0

            idx_invalidfield = idx_invalidfield | idx_invalidfield3
            self.np_pieces[idx_invalidfield, IDX_VALIDMOVE] = 1  # already covered fields are not a valid move


        else:
            print(self.np_pieces[row, column, IDX_VALIDMOVE])
            raise ValueError("Can't play this field on board. Check for valid moves.")

    def check_game_ended(self):
        """The game ends if all marbles are placed on board or if the next player has no marble"""

        if np.sum(self.np_pieces[:, :, IDX_FIELDSTATE] == PLAYERS[0]) >= 28:
            return True

        elif np.sum(self.np_pieces[:, :, IDX_FIELDSTATE] == PLAYERS[1]) >= 28:
            return True

        elif np.all(self.np_pieces[:, :, IDX_VALIDMOVE]):
            return True

        else:
            return False

    def get_winner(self):
        """Identify winner of the game"""

        "Calculate the points of every player"
        "Sortieren der Holzplatten"
        "Welche Holzplatten gibt es (Größe und Anzahl)"
        "Doppelte for Schleife"
        "Init points of players"
        points_player1 = 0
        points_player2 = 0

        "Get plates of board"
        #plates = np.unique(self.np_pieces[:, :, IDX_SIZEPLATE])
        for sizeplate in np.unique(self.np_pieces[:, :, IDX_SIZEPLATE]):

            idx_sizeplate = np.where(self.np_pieces[:, :, IDX_SIZEPLATE] == sizeplate)

            for numplate in np.unique(self.np_pieces[idx_sizeplate[0], idx_sizeplate[1], IDX_NUMPLATE]):

                "Get number of fields of each player per plate. The player who has more marbles on a plate, " \
                "gets points. The points equal the size of the plate."

                idx_plate = ((self.np_pieces[:, :, IDX_SIZEPLATE] == sizeplate) & (self.np_pieces[:, :, IDX_NUMPLATE] == numplate))
                "Count fields of player 1"
                fields_player1 = np.count_nonzero(self.np_pieces[idx_plate, IDX_FIELDSTATE] == PLAYERS[0])
                fields_player2 = np.count_nonzero(self.np_pieces[idx_plate, IDX_FIELDSTATE] == PLAYERS[1])

                if fields_player1 > fields_player2:
                    points_player1 = points_player1 + sizeplate
                elif fields_player1 < fields_player2:
                    points_player2 = points_player2 + sizeplate

        if points_player1 > points_player2:
            return [PLAYERS[0], points_player1, points_player2]
        elif points_player1 < points_player2:
            return [PLAYERS[1], points_player2, points_player1]
        else:
            return [0, 0, 0]