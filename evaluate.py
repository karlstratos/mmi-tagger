import numpy as np
from collections import Counter


def compute_many2one_acc(tseqs, zseqs):
    mapping = get_majority_mapping(tseqs, zseqs)

    num_instances = 0
    num_correct = 0
    for i in range(len(tseqs)):
        for (t, z) in zip(tseqs[i], zseqs[i]):
            num_instances += 1
            if mapping[z] == t:
                num_correct += 1
    acc = num_correct / num_instances * 100

    return acc


def compute_v_measure(tseqs, zseqs):
    num_instances = 0
    t2i = {}
    z2i = {}
    cocount = Counter()
    for i in range(len(tseqs)):
        for (t, z) in zip(tseqs[i], zseqs[i]):
            num_instances += 1
            if not t in t2i: t2i[t] = len(t2i)
            if not z in z2i: z2i[z] = len(z2i)
            cocount[(t2i[t], z2i[z])] += 1

    B = np.empty([len(t2i), len(z2i)])
    for i in range(len(t2i)):
        for j in range(len(z2i)):
            B[i, j] = cocount[(i, j)] / num_instances

    p_T = np.sum(B, axis=1)
    p_Z = np.sum(B, axis=0)
    H_T = sum([- p_T[i] * np.log2(p_T[i]) for i in range(len(t2i))])
    H_Z = sum([- p_Z[i] * np.log2(p_Z[i]) for i in range(len(z2i))])

    H_T_given_Z = 0
    for j in range(len(z2i)):
        for i in range(len(t2i)):
            if B[i, j] > 0.0:
                H_T_given_Z -= B[i, j] * \
                               (np.log2(B[i, j]) - np.log2(p_Z[j]))
    H_Z_given_T = 0
    for j in range(len(t2i)):
        for i in range(len(z2i)):
            if B[j, i] > 0.0:
                H_Z_given_T -= B[j, i] * \
                               (np.log2(B[j, i]) - np.log2(p_T[j]))

    h = 1 if len(t2i) == 1 else 1 - H_T_given_Z / H_T
    c = 1 if len(z2i) == 1 else 1 - H_Z_given_T / H_Z

    return 2 * h * c / (h + c) * 100.0


def get_majority_mapping(tseqs, zseqs):
    cooccur = count_cooccurence(tseqs, zseqs)
    mapping = {}
    for z in cooccur:
        mapping[z] = max(cooccur[z].items(), key=lambda x: x[1])[0]
    return mapping


def count_cooccurence(tseqs, zseqs):
    cooccur = {}
    assert len(tseqs) == len(zseqs)
    for i in range(len(tseqs)):
        assert len(tseqs[i]) == len(zseqs[i])
        for (t, z) in zip(tseqs[i], zseqs[i]):
            if not z in cooccur: cooccur[z] = Counter()
            cooccur[z][t] += 1
    return cooccur
