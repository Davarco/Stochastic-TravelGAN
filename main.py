import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Configuration details for training and testing Stochastic TravelGAN.')
parser.add_argument('--train', action='store_true')
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

def main():
    with tf.Session() as sess:
        print("Hello")

if __name__ == "__main__":
    main()
