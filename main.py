from argparse import ArgumentParser
from IQLAgent import IQLAgent

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--env_name', '-e',type=str, default='ant-expert-v2')
    parser.add_argument('--gradient_step', '-g', type=int, default=500000)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    parser.add_argument('--learning_rate', '-l', type=float, default=3e-4)
    parser.add_argument('--discount_factor', '-d', type=float, default=0.99)
    parser.add_argument('--tau', '-t', help='expectile', type=float, default=0.7)
    parser.add_argument('--alpha', '-a', help='soft update ratio', type=float, default=5e-3)
    args = parser.parse_args()
    
    # train
    agent = IQLAgent(args)
    agent.train()