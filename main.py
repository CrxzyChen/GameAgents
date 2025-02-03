import argparse
import os

import yaml

from uilts.loader import module_instance

parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, help='Name to greet')
parser.add_argument('--test', type=str, help='Name to greet')


def main():
    args = parser.parse_args()
    if args.train is not None:
        if os.path.exists(args.train):
            configs = yaml.load(open(args.train, 'r'), Loader=yaml.FullLoader)
            trainer_config = configs.get('trainer', {})
            train_config = configs.get('train', {})
            if trainer_config is not None:
                trainer = module_instance(trainer_config)
                trainer.train(**train_config)
            else:
                print('Trainer config not found')
        else:
            print(f'File {args.train} not found')
    elif args.test is not None:
        print(f'Test {args.test}')
        if os.path.exists(args.test):
            configs = yaml.load(open(args.test, 'r'), Loader=yaml.FullLoader)
            trainer_config = configs.get('trainer', {})
            test_config = configs.get('test', {})
            if trainer_config is not None:
                trainer = module_instance(trainer_config)
                trainer.test(**test_config)
            else:
                print('Tester config not found')
    else:
        print('No command found')


if __name__ == '__main__':
    main()
