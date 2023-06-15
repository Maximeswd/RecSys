"""
Codes for running the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
import yaml

import tensorflow as tf

from trainer import Trainer

parser = argparse.ArgumentParser()
possible_model_names = ['bpr', 'ubpr', 'wmf', 'expomf', 'relmf']
parser.add_argument('--model_name', '-m', type=str, required=True, choices=possible_model_names)
parser.add_argument('--run_sims', '-r', type=int, default=10, required=True)
parser.add_argument('--data', '-d', type=str, required=True, choices=['coat', 'yahoo'])
parser.add_argument('--pointwise_loss', type=str, help='Pointwise loss function to use', required=True)
parser.add_argument('--pairwise_loss', type=str, help='Pairwise loss function to use', required=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    config = yaml.safe_load(open('../conf/config.yaml', 'rb'))
    trainer = Trainer(
        data=args.data,
        batch_size=config['batch_size'],
        max_iters=config['max_iters'],
        eta=config['eta'],
        model_name=args.model_name,
        pointwise_loss=args.pointwise_loss,
        pairwise_loss=args.pairwise_loss
    )
    trainer.run(num_sims=args.run_sims)

    # debug variant
    # warnings.filterwarnings("ignore")
    # tf.get_logger().setLevel("ERROR")
    
    # config = yaml.safe_load(open('recsys/conf/config.yaml', 'rb'))
    # trainer = Trainer(
    #     data='yahoo',
    #     batch_size=config['batch_size'],
    #     max_iters=config['max_iters'],
    #     eta=config['eta'],
    #     model_name='wmf',
    #     pointwise_loss='dual_unbiased_loss',
    #     pairwise_loss='paper_loss'
    # )
    # trainer.run(num_sims=1)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running !')
    print('\n', '=' * 25, '\n')
