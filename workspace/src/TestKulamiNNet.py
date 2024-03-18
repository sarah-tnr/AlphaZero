import torch
import pytest
from KulamiNNet import KulamiNNet
from KulamiGame import KulamiGame

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

@pytest.fixture
def nnet():
    game = KulamiGame()
    # Create a KulamiNNet instance
    nnet = KulamiNNet(game, args)

    # Set the input dimensions
    nnet.board_x = 8
    nnet.board_y = 8
    nnet.board_z = 6

    # Set the arguments
    nnet.args = torch.nn.Module()
    nnet.args.num_channels = 64
    nnet.args.dropout = 0.5

    return nnet


def test_kulami_nnet_forward_shape(nnet):
    batch_size = 16
    input_tensor = torch.randn(batch_size, nnet.board_x, nnet.board_y, nnet.board_z)

    pi, v = nnet.forward(input_tensor)

    assert pi.shape == torch.Size([batch_size, nnet.action_size])
    assert v.shape == torch.Size([batch_size, 1])


def test_kulami_nnet_forward_probabilities(nnet):
    batch_size = 16
    input_tensor = torch.randn(batch_size, nnet.board_x, nnet.board_y, nnet.board_z)

    pi, _ = nnet.forward(input_tensor)

    probabilities = torch.exp(pi)
    assert torch.allclose(torch.sum(probabilities, dim=1), torch.ones(batch_size))


def test_kulami_nnet_forward_v_range(nnet):
    batch_size = 16
    input_tensor = torch.randn(batch_size, nnet.board_x, nnet.board_y, nnet.board_z)

    _, v = nnet.forward(input_tensor)

    assert torch.all(v >= -1) and torch.all(v <= 1)


def test_kulami_nnet_forward_dropout(nnet):
    batch_size = 16
    input_tensor = torch.randn(batch_size, nnet.board_x, nnet.board_y, nnet.board_z)

    pi, _ = nnet.forward(input_tensor)

    pi_dropout = nnet._apply_dropout(pi)
    assert torch.allclose(pi_dropout, F.dropout(pi, p=nnet.args.dropout, training=True))


def test_kulami_nnet_forward_relu(nnet):
    batch_size = 16
    input_tensor = torch.randn(batch_size, nnet.board_x, nnet.board_y, nnet.board_z)

    pi, v = nnet.forward(input_tensor)

    assert torch.all(pi >= 0) and torch.all(v >= 0)


if __name__ == "__main__":
    pytest.main()
