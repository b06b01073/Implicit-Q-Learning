from argparse import ArgumentParser
from IQLAgent import IQLAgent

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--env_name', '-e',type=str, default='walker2d-medium-expert-v2')
    parser.add_argument('--gradient_step', '-g', type=int, default=500000)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    parser.add_argument('--learning_rate', '-l', type=float, default=3e-4)
    parser.add_argument('--discount_factor', '-d', type=float, default=0.99)
    parser.add_argument('--tau', '-t', help='expectile', type=float, default=0.7)
    parser.add_argument('--alpha', '-a', help='soft update ratio', type=float, default=5e-3)
    parser.add_argument('--beta', help='actor temperature', type=float, default=3.0)
    parser.add_argument('--max_adv', help='maximum weight of exponentiated advantage', type=float, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--normalize_score', action='store_false')
    parser.add_argument('--write_file', type=str, default='result.txt')

    args = parser.parse_args()
    
    # train
    agent = IQLAgent(args)
    agent.train()
    agent.evaluate()
