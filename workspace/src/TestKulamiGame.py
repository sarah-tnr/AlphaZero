import numpy as np
import pytest
from KulamiGame import KulamiGame

# Testen Sie die getInitBoard-Methode
def test_getInitBoard():
    game = KulamiGame()
    board = game.getInitBoard()

    # Überprüfen Sie, ob das zurückgegebene Board korrekt ist
    assert np.all(board.np_pieces == game.board.np_pieces)

# Testen Sie die getBoardSize-Methode
def test_getBoardSize():
    game = KulamiGame()
    size = game.getBoardSize()

    # Überprüfen Sie, ob die zurückgegebene Größe korrekt ist
    assert size == game.board.np_pieces.shape

# Testen Sie die getActionSize-Methode
def test_getActionSize():
    game = KulamiGame()
    actionSize = game.getActionSize()

    # Überprüfen Sie, ob die zurückgegebene Anzahl der Aktionen korrekt ist
    assert actionSize == game.board.np_pieces.shape[0] * game.board.np_pieces.shape[1]

# Testen Sie die getNextState-Methode
def test_getNextState():
    game = KulamiGame()
    board = game.getInitBoard()
    player = 1
    action = (0, 0)
    nextState, nextPlayer = game.getNextState(board, player, action)

    # Überprüfen Sie, ob das nächste Board und der nächste Spieler korrekt sind
    assert np.all(nextState == game.board.np_pieces)
    assert nextPlayer == -player

# Testen Sie die getValidMoves-Methode
def test_getValidMoves():
    game = KulamiGame()
    board = game.getInitBoard()
    player = 1
    validMoves = game.getValidMoves(board, player)

    # Überprüfen Sie, ob die gültigen Züge korrekt sind
    assert np.all(validMoves == game.board.get_valid_moves())

# Testen Sie die getGameEnded-Methode
def test_getGameEnded():
    game = KulamiGame()
    board = game.getInitBoard()
    player = 1
    gameEnded = game.getGameEnded(board, player)

    # Überprüfen Sie, ob das Spiel korrekt beendet wurde
    assert gameEnded == game.board.check_game_ended()


# Testen der getCanonicalForm-Methode
def test_getCanonicalForm():
    # Ein neues KulamiGame-Objekt erstellen
    game = KulamiGame()
    # Ein Testbrett erstellen
    test_board = game.getInitBoard()
    #test_board.np_pieces = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    # Überprüfen, ob das Ergebnis der Methode korrekt ist, wenn player == 1
    assert np.all(game.getCanonicalForm(test_board, 1) == test_board.np_pieces[:, :, test_board.idxFieldState])
    # Überprüfen, ob das Ergebnis der Methode korrekt ist, wenn player == -1
    assert np.all(game.getCanonicalForm(test_board, -1) == -test_board.np_pieces[:, :, test_board.idxFieldState])
