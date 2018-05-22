import os
import sys
import random
import glob
import shutil
import pickle
import operator
import warnings
from multiprocessing import Pool

import pandas as pd
import numpy as np

from tqdm import tqdm


def build_ignored():
    return set()


def to_id(fname):
    return os.path.basename(fname).split('.')[0]


def to_fname(image_id):
    return os.path.join('/opt/kaggle/landmark-recognition/data/train-copy',
                        image_id[:3], "{}.jpg".format(image_id))


def queryid_fname(image_id):
    return os.path.join('/kaggle1/landmark-recognition/test',
                        image_id[:3], "{}.jpg".format(image_id))


def build_landmarks():
    df = pd.read_csv('/kaggle1/landmark-recognition/train.csv')
    id_to_landmark = df[['id', 'landmark_id']].set_index('id').to_dict()['landmark_id']
    return id_to_landmark


def delf_path(image_id, kind='train'):
    base_path = "/opt/kaggle/landmark-recognition/data/delf-desc/{}".format(kind)
    return os.path.join(base_path, image_id[:3], '%s.npz' % (image_id))


def ransac_geometric_validation(fpath1, fpath2, min_samples=3):
    """ Count inliers
    """

    from scipy.spatial import cKDTree
    from skimage.measure import ransac
    from skimage.transform import AffineTransform

    _DISTANCE_THRESHOLD = 0.8

    if not os.path.exists(fpath1):
        print('*** not found: ', fpath1)
        return 0

    if not os.path.exists(fpath2):
        print('*** not found: ', fpath2)
        return 0

    x1 = np.load(fpath1)
    locations_1, descriptors_1 = x1['locations'], x1['desc']
    num_features_1 = locations_1.shape[0]

    x2 = np.load(fpath2)
    locations_2, descriptors_2 = x2['locations'], x2['desc']
    num_features_2 = locations_2.shape[0]

    d1_tree = cKDTree(descriptors_1)
    distances, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    if locations_1_to_use.shape[0] < min_samples \
            or locations_2_to_use.shape[0] < min_samples:
        return 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Perform geometric verification using RANSAC.
        model_robust, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=min_samples,
            residual_threshold=20,
            max_trials=10)
    x = np.sum(inliers)
    if x is None:
        return 0
    else:
        return x


def ransac(pair):
    try:
        count = ransac_geometric_validation(pair[0], pair[1])
        return count
    except Exception as e:
        print(e)
        return 0

def reco_score():
    landmarks = build_landmarks()

    with open('reco_fnames.pkl', 'rb') as f:
        fnames = pickle.load(f)

    scores = np.load('reco_nearest.npy')
    idx = np.load('reco_nearest_idx.npy')

    images, qimages = fnames
    image_ids = [to_id(x) for x in images]
    qimage_ids = [to_id(x) for x in qimages]
    ignore_ids = build_ignored()

    ignored = set()
    exceptions = 0
    result = {}

    pool = Pool(8)

    for i, image_id in tqdm(enumerate(qimage_ids), total=len(qimage_ids)):
        if image_id in ignore_ids:
            ignored.add(image_id)
            continue

        matches = [image_ids[x] for x in idx[i, :]]
        distances = scores[i, :]

        if distances[0] < 0.01:
            # suspicious
            for match_id, match_distance in zip(matches, distances):
                print('*** ', image_id, match_id, match_distance)
                shutil.copy(to_fname(match_id), '/tmp/tmp1')
                shutil.copy(queryid_fname(image_id), '/tmp/tmp1')
                return

        # perform geometric validation on top_n matches
        top_n = 8
        use_ransac = True
        inliers = {}
        distance = {}
        for m, d in zip(matches[:top_n], distances[:top_n]):
            distance[m] = d

        pairs = []
        for match_id in matches[:top_n]:
            fname1 = delf_path(image_id, kind='test')
            fname2 = delf_path(match_id, kind='train')
            pairs.append((fname1, fname2))

        pairs_inliers = pool.map(ransac, pairs)
        for match_id, count in zip(matches[:top_n], pairs_inliers):
            inliers[match_id] = -1 * count

        if use_ransac:
            inliers = sorted(inliers.items(), key=operator.itemgetter(1))
            k = inliers[0][0]
            v = inliers[0][1]
            d = distance[k]
            # result: landmark_id, number of inliers for top match, distance
            result[image_id] = [landmarks[k], v, d]
        else:
            result[image_id] = [None, None, None]

        # grouping by landmark -- idea didn't work well
        if False:
            by_landmark = {}

            for m, d in zip(matches, distances):
                if m in ignore_ids:
                    continue
                landmark = landmarks[m]
                if landmark not in by_landmark:
                    by_landmark[landmark] = []
                by_landmark[landmark].append(d)

            avg_distance_by_landmark = {}
            for landmark, values in by_landmark.items():
                avg_distance_by_landmark[landmark] = np.sum(values)/len(values)

            sorted_landmarks = sorted(avg_distance_by_landmark.items(), key=operator.itemgetter(1))
            r = ','.join(["{}:{}".format(k, v) for k, v in sorted_landmarks[:10]])

            best = sorted_landmarks[0][0]
            best_score = 1 - sorted_landmarks[0][1]
            # print("{},{} {:.4f}".format(image_id, best, best_score))
            result[image_id] = "{} {:.4f}".format(best, best_score)

    # make up some scoring heuristic
    # inliers count descending and cnn distance ascending
    sorted_result = sorted(result.items(), key=lambda x: (x[1][1], x[1][2]))
    scores = {}
    for idx, (k, v) in enumerate(sorted_result):
        scores[k] = 1 - (idx / len(sorted_result))

    print('ignored: ', len(ignored))

    rows = []
    debug = []
    ids = pd.read_csv('/opt/kaggle/landmark-recognition/input/sample_submission.csv', usecols=['id']).id.values
    for image_id in qimage_ids:
        if image_id in result:
            landmark_id, inliers, distance = result[image_id]
            score = scores[image_id]
            rows.append([image_id, "{} {:.4f}".format(landmark_id, score)])
            debug.append([image_id, "{} {} {} {}".format(landmark_id, inliers, distance, score)])
        else:
            rows.append([image_id, ""])

    pd.DataFrame(rows, columns=['id', 'landmarks']).to_csv('reco_subm.csv', index=False)
    # pd.DataFrame(debug, columns=['id', 'debug']).to_csv('reco_debug.csv', index=False)


if __name__ == '__main__':
    reco_score()
