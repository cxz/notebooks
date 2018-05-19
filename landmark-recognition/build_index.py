""" Build Index with CNN activations

https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors
https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

nvidia-docker run -ti --name retr-faiss -v /opt/kaggle/landmark-retrieval:/landmark faiss bash
PYTHONPATH=/opt/faiss python build_index.py

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


def build_index(index_name):
    start = time.time()
    vecs = np.load("reco_vecs.npy")
    vecs_t = vecs.T
    print('vecs:', vecs_t.shape, vecs.dtype)

    d = vecs_t.shape[1]
    index = faiss.index_factory(d, index_name)

    train_idx = np.random.choice(vecs.shape[1], 200000)
    xt = vecs_t[train_idx, :]
    index.train(xt)
    print('index trained: ', htime(time.time() - start))

    chunk_size = 100000
    for chunk_idx, chunk in enumerate(chunks(range(vecs_t.shape[0]), chunk_size)):
        print('chunk: ', chunk_idx, 1 + vecs_t.shape[0]//chunk_size)
        x = vecs_t[chunk, :]
        index.add(x)

    faiss.write_index(index, "reco.index")
    print('done: ', htime(time.time() - start))


if __name__ == '__main__':
    build_index('IVF4096,Flat')
