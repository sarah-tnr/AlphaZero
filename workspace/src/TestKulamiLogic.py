import numpy as np
import pytest

#from ..Environment.KulamiLogic import Board
from KulamiLogic import Board
# Test fixture
@pytest.fixture
def empty_board():
    return Board()

# Test fixture with a pre-filled board
@pytest.fixture
def filled_board():
    np_pieces = np.zeros([8, 8], dtype=np.dtype('(6,)i4'))
    np_pieces[0, 0, :3] = (4, 1, 1)
    np_pieces[0, 1, :3] = (4, 1, 2)
    # Add more pre-filled pieces here...
    return Board(np_pieces=np_pieces)

def test_board_creation(empty_board):
    # Test that an empty board is created correctly
    assert empty_board.height == 8
    assert empty_board.width == 8
    assert empty_board.np_pieces.shape == (8, 8, 6)

def test_board_add_stone(empty_board):
    # Test adding a stone to an empty board
    empty_board.add_stone(0, 0, 1)
    assert empty_board.np_pieces[0, 0, 3] == 1  # Check that the field state is updated
    assert empty_board.np_pieces[0, 0, 4] == 1  # Check that the move count is updated

def test_board_choose_player(empty_board):
    # Test that the player is chosen correctly
    assert empty_board.choose_player() in [-1, 1]

def test_board_get_valid_moves(empty_board):
    # Test that all fields are initially valid moves
    assert np.all(empty_board.get_valid_moves() == 0)

def test_board_check_game_ended(empty_board):
    # Test that the game is not ended initially
    assert empty_board.check_game_ended() == False

def test_board_get_winner(empty_board):
    # Test that there is no winner initially
    assert empty_board.get_winner() == [0, 0, 0]

def test_simulate_game(empty_board):
    # Test that a game can be simulated
    while not empty_board.check_game_ended():
        player = empty_board.choose_player()
        valid_moves = empty_board.get_valid_moves()
        valid_move_indices = np.where(valid_moves == 0)
        random_index = np.random.choice(len(valid_move_indices[0]))
        row, column = valid_move_indices[0][random_index], valid_move_indices[1][random_index]
        empty_board.add_stone(row, column, player)


    winner = empty_board.get_winner()
    print(winner)

    assert winner[0] in [-1, 1]
    assert winner[1] >= 0
    assert winner[2] >= 0