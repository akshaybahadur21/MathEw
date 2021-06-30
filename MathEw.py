import argparse

from src.MathewApp import MathewApp
from src.MathewTrainer import MathewTrainer
from src.utils.utils import str2bool


class MathEw:
    def __init__(self):
        self.mathew_trainer = MathewTrainer()
        self.mathew_app = MathewApp()

    def performCalulation(self, train):
        if train:
            self.mathew_trainer.train()
        self.mathew_app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', "--train", required=True, help="Train the model or not",
                        type=str2bool)
    args = parser.parse_args()
    train = args.train
    mathew = MathEw()
    mathew.performCalulation(train)
