from features.build_features import build_features
from models.train import train
from models.tune import tune
def generate_model():
    build_features()
    train()
    tune()


if __name__== '__main__':
    generate_model()