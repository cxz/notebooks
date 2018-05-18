""" Extract vectors.
"""

import argparse
import os
import time
import math
import pickle

import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

def landmark_recognition_dataset():
    base_path = '/kaggle1/landmark-recognition'
    train_ids = pd.read_csv(os.path.join(base_path, 'train.csv'), usecols=['id']).id.values
    test_ids = pd.read_csv(os.path.join(base_path, 'test.csv'), usecols=['id']).id.values
    
    # base path where images are stored
    base_train = '/opt/kaggle/landmark-recognition/data/train-copy'
    base_test = os.path.join(base_path, 'test')
    
    # database images
    images = [os.path.join(base_train, image_id[:3], '%s.jpg' % image_id) for image_id in train_ids]
    images = [fpath for fpath in images if os.path.exists(fpath)]
        
    # query images
    qimages = [os.path.join(base_test, image_id[:3], '%s.jpg' % image_id) for image_id in test_ids]
    qimages = [fpath for fpath in qimages if os.path.exists(fpath)]
    
    print('== landmark recognition dataset ===')
    print('database:', len(train_ids), len(images))
    print('query:', len(test_ids), len(qimages))
    return images, qimages
    

datasets_names = ['oxford5k,paris6k', 'roxford5k,rparis6k', 'oxford5k,paris6k,roxford5k,rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help='network path, destination where network is saved')
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help='network off-the-shelf, in the format ARCHITECTURE-POOLING or ARCHITECTURE-POOLING-whiten,' + 
                    ' examples: resnet101-gem | resnet101-gem-whiten')

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                   help='comma separated list of test datasets: ' + 
                        ' | '.join(datasets_names) + 
                        ' (default: oxford5k,paris6k)')
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help='maximum size of longer image side used for testing (default: 1024)')
parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true',
                    help='use multiscale vectors for testing')
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help='dataset used to learn whitening for testing: ' + 
                        ' | '.join(whitening_names) + 
                        ' (default: None)')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for testing (default: 0)')

def main():
    args = parser.parse_args()

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:
        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        state = torch.load(args.network_path)
        net = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'], 
                            mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
        net.load_state_dict(state['state_dict'])
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        offtheshelf = args.network_offtheshelf.split('-')
        if len(offtheshelf)==3:
            if offtheshelf[2]=='whiten':
                offtheshelf_whiten = True
            else:
                raise(RuntimeError("Incorrect format of the off-the-shelf network. Examples: resnet101-gem | resnet101-gem-whiten"))
        else:
            offtheshelf_whiten = False
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(model=offtheshelf[0], pooling=offtheshelf[1], whitening=offtheshelf_whiten)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        print('>> {}: Learning whitening...'.format(args.whitening))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', args.whitening)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        # extract whitening vectors
        print('>> {}: Extracting...'.format(args.whitening))
        wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        
        # learning whitening 
        print('>> {}: Learning...'.format(args.whitening))
        wvecs = wvecs.numpy()
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        if dataset == 'reco':
            images, qimages = landmark_recognition_dataset()
            bbxs = [None for x in qimages]

        else:
            # prepare config structure for the test dataset
            cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
            images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        with open('%s_fnames.pkl' % dataset, 'wb') as f:
            pickle.dump([images, qimages], f)

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        vecs = vecs.numpy()
        print('>> saving')
        np.save('{}_vecs.npy'.format(dataset), vecs)
        
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        qvecs = qvecs.numpy()
        np.save('{}_qvecs.npy'.format(dataset), qvecs)
        
            
        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            
            # TODO
            
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
