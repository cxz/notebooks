""" Build Index with CNN activations

https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors
https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

nvidia-docker run -ti --name retr-faiss -v /opt/kaggle/landmark-retrieval:/landmark faiss bash
PYTHONPATH=/opt/faiss python search.py

"""
import time
import numpy as np
import faiss


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def htime(c):
    c = round(c)

    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{}d {}h {}m {}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{}h {}m {}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{}m {}s'.format(minutes, seconds)
    return '{}s'.format(seconds)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def search(dataset, query_ds, out_prefix='', k_nearest=1000, nprobe=10):
    start = time.time()

    index = faiss.read_index('%s.index' % dataset)
    index.nprobe = nprobe
    print('index loaded')

    # same queries for both reco & retr datasets
    x_queries = np.load(query_ds).T

    nearest = np.zeros((x_queries.shape[0], k_nearest), dtype=np.int64)
    scores = np.zeros((x_queries.shape[0], k_nearest), dtype=np.float32)

    chunk_size = 1000
    for chunk_idx, chunk in enumerate(chunks(range(x_queries.shape[0]), chunk_size)):
        print(chunk_idx, x_queries.shape[0]//chunk_size)
        D, I = index.search(x_queries[chunk, :], k_nearest)
        print(np.min(D), np.max(D), np.min(I), np.max(I))
        nearest[chunk, :] = I
        scores[chunk, :] = D

    np.save("%s%s_nearest.npy" % (dataset, out_prefix), scores)
    np.save("%s%s_nearest_idx.npy" % (dataset, out_prefix), nearest)

    print('done: ', htime(time.time() - start))


if __name__ == '__main__':
    # search('reco', 'reco_qvecs.npy', 'reco')
    # search('retr', 'reco_qvecs.npy', 'retr')
    # search('reco', 'reco_qvecs_val.npy', '_val')
    search('reco', 'reco_qvecs_val_junk.npy', '_val_junk')
