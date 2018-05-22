import glob
import os
import pickle

import pandas as pd
import numpy as np


def build_ignored():
    return set()


def to_id(fname):
    return os.path.basename(fname).split('.')[0]


def queryid_fname(image_id):
    # same query images as landmark-recognition
    return os.path.join('/kaggle1/landmark-recognition/test',
                        image_id[:3], "{}.jpg".format(image_id))


def retr_score(ids):
    with open('reco_fnames.pkl', 'rb') as f:
        reco_fnames = pickle.load(f)
        _, qimages = reco_fnames

    with open('retr_fnames.pkl', 'rb') as f:
        images, _ = pickle.load(f)

    image_ids = [to_id(x) for x in images]
    qimage_ids = [to_id(x) for x in qimages]
    ignore_ids = build_ignored()

    scores = np.load('retr_nearest.npy')
    idx = np.load('retr_nearest_idx.npy')
    # print(scores.shape)
    # print(idx.shape)

    print('images: ', len(image_ids),
          ' qimages: ', len(qimage_ids),
          ' ignored: ', len(ignore_ids))

    result = {}

    for i, image_id in enumerate(qimage_ids):
        if image_id in ignore_ids:
            continue

        matches = [image_ids[x] for x in idx[i, :]]
        distances = scores[i, :]

        if distances[0] < 0.01:
            # suspicious
            for match_id, match_distance in zip(matches, distances):
                print('*** ', image_id, match_id, match_distance)
                # shutil.copy(to_fname(match_id), '/tmp/tmp1')
                # shutil.copy(queryid_fname(image_id), '/tmp/tmp2')

        row_str = " ".join(["{}".format(k) for k, v in zip(matches, distances)])
        result[image_id] = row_str

    rows = []
    for image_id in ids:
        if image_id in result:
            row_str = result[image_id]
        else:
            row_str = ""
        rows.append([image_id, row_str])

    df = pd.DataFrame(rows, columns=['id', 'images'])
    df.to_csv('retr_subm.csv', index=False)


if __name__ == '__main__':
    ids = pd.read_csv('/opt/kaggle/landmark-retrieval/input/sample_submission.csv', usecols=['id']).id.values
    retr_score(ids)
