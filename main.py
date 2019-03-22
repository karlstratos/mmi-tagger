import argparse
import codecs
import random
import os
import sys
import torch

from control import Control
from data import Data
from logger import Logger
from model import MMIModel


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    data = Data(args.data)
    model = MMIModel(len(data.w2i), len(data.c2i), args.num_labels, args.dim,
                     args.dim // 2, args.width, 1).to(device)
    logger = Logger(args.model + '.log', args.train)
    logger.log('python ' + ' '.join(sys.argv) + '\n')
    logger.log('Random seed: %d' % args.seed)
    control = Control(model, args.model, args.batch_size, device, logger)

    if args.train:
        acc, vm, zseqs, clustering = control.train(data, args.lr, args.epochs)

    elif os.path.exists(args.model):
        control.load_model()
        acc, vm, zseqs, clustering = control.evaluate(data)
        print('     acc: {:5.2f}'.format(acc))
        print('      vm: {:5.2f}'.format(vm))

    if args.pred:
        with open(args.pred, 'w') as f:
            for zseq in zseqs:
                f.write(' '.join([str(z) for z in zseq]) + '\n')

    if args.clusters:
        with open(args.clusters, 'w') as f:
            for z, cluster in enumerate(clustering):
                f.write(str(z) + '\t' + ' '.join([data.i2w[i] for i in
                                                  cluster]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maximal Mutual Information (MMI) Tagger')

    parser.add_argument('model', type=str,
                        help='model path')
    parser.add_argument('data', type=str,
                        help='data path (X.words, assumes X.tags exists)')
    parser.add_argument('--num_labels', type=int, default=45, metavar='m',
                        help='number of labels to induce [%(default)d]')
    parser.add_argument('--train', action='store_true',
                        help='train?')
    parser.add_argument('--batch_size', type=int, default=80, metavar='B',
                        help='batch size [%(default)d]')
    parser.add_argument('--dim', type=int, default=200,
                        help='dimension of word embeddings [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--width', type=int, default=2,
                        help='context width (to each side) [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--pred', type=str, default='',
                        help='prediction path')
    parser.add_argument('--clusters', type=str, default='',
                        help='cluster path')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')

    args = parser.parse_args()
    main(args)
