import glob
import os
import pickle
import csv
import operator
import warnings
from multiprocessing import Pool

import pandas as pd
import numpy as np

from tqdm import tqdm


def to_id(fname):
    return os.path.basename(fname).split('.')[0]


def queryid_fname(image_id):
    # same query images as landmark-recognition
    return os.path.join('/kaggle1/landmark-recognition/test',
                        image_id[:3], "{}.jpg".format(image_id))


def delf_path(image_id, kind='train'):
    base_path = "/opt/kaggle/landmark-recognition/data/delf-desc/{}".format(kind)
    return os.path.join(base_path, image_id[:3], '%s.npz' % (image_id))


def retr_delf_path(image_id):
    base_path = "/opt/kaggle/landmark-retrieval/data-train-delf-desc"
    return os.path.join(base_path, image_id[:3], '%s.npz' % image_id)


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
            max_trials=50)
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


def run_geometric_validation(pool, image_id, matches, distances, top_k=10, top_n=100):
    inliers = {}

    # candidate pairs to run ransac on
    pairs = []
    for match_id in matches[:top_k]:
        fname1 = delf_path(image_id, kind='test')
        fname2 = retr_delf_path(match_id)
        pairs.append((fname1, fname2))

    pairs_inliers = pool.map(ransac, pairs)
    for match_id, count in zip(matches[:top_k], pairs_inliers):
        inliers[match_id] = -1 * count

    # sort by number of inliers
    inliers = sorted(inliers.items(), key=operator.itemgetter(1))

    result = [x[0] for x in inliers]
    result.extend(matches[top_k:])  # add the rest we couldn't verify -- already sorted by distance
    return result


def retr_score(ids):
    with open('reco_fnames.pkl', 'rb') as f:
        reco_fnames = pickle.load(f)
        _, qimages = reco_fnames

    with open('retr_fnames.pkl', 'rb') as f:
        images, _ = pickle.load(f)

    image_ids = [to_id(x) for x in images]
    qimage_ids = [to_id(x) for x in qimages]

    scores = np.load('retr_nearest.npy')
    idx = np.load('retr_nearest_idx.npy')
    # print(scores.shape)
    # print(idx.shape)

    print('images: ', len(image_ids), ' qimages: ', len(qimage_ids))

    pool = Pool(4)

    result = {}
    debug = {}

    for i, image_id in tqdm(enumerate(qimage_ids), total=len(qimage_ids)):

        matches = [image_ids[x] for x in idx[i, :]]
        distances = scores[i, :]

        if distances[0] < 0.01:
            # suspicious --
            continue

        # reorder matches found using global descriptors
        # according to number of inliers in geometric validation
        min_distance = distances[0]

        if min_distance < 0.3:
            # arbitrary threshold, not enough power to run on full dataset
            top_matches = run_geometric_validation(pool, image_id, matches, distances)
            row_str = " ".join(["{}".format(k) for k in top_matches])
        else:
            top_100 = zip(matches[:100], distances[:100])
            row_str = " ".join(["{}".format(k) for k, v in top_100])

        result[image_id] = row_str
        debug[image_id] = distances[0]

    rows = []
    rows_debug = []
    for image_id in ids:
        if image_id in result:
            row_str = result[image_id]
            debug_str = debug[image_id]
        else:
            row_str = ""
            debug_str = ""
        rows.append([image_id, row_str])
        rows_debug.append([image_id, debug_str])

    df = pd.DataFrame(rows, columns=['id', 'images'])
    df.to_csv('retr_subm.csv', index=False)

    pd.DataFrame(rows_debug, columns=['id', 'min_distance'])\
        .to_csv('retr_debug.csv', index=False)


def delf_matches():
    h = {}  # image -> match dict
    with open('retr-delf-k1000.0', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader, total=117689):
            matches = {}  # match_id -> inlier_count
            matches_values = row[1].split(' ')
            image_id = row[0]
            for k, v in [v.split(':') for v in matches_values if len(v) > 0]:
                matches[k] = v
            h[image_id] = matches
    return h


if __name__ == '__main__':
    # delf_matches()
    ids = pd.read_csv('/opt/kaggle/landmark-retrieval/input/sample_submission.csv', usecols=['id']).id.values
    retr_score(ids)
