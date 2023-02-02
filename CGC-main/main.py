import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
import warnings
from Adam import adam
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
warnings.filterwarnings("ignore")

def Run(data):
    # Load data
    if(dataset == 'large_cora'):
        X=data['X']
        X = X.toarray()
        A = data['G']
        A = A.toarray()
        gnd = data['labels']
        gnd = gnd[0, :]

    else:
        X = data['fea']
        if(type(X) is np.ndarray):
            pass
        else:
            X = X.toarray()
        A = data['W']
        if(type(A) is np.ndarray):
            pass
        else:
            A = A.toarray()
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]

    # Store some variables
    N = X.shape[0]
    k = len(np.unique(gnd))
    I = np.eye(N)
    if sp.issparse(X):
        X = X.todense()

    # Normalize A
    A = A + I
    D = np.sum(A,axis=1)
    D = np.power(D,-0.5)
    D[np.isinf(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)
    Ls = I - A

    # Set Papameters
    kk = 1
    a = 10
    lambda_ = 10

    #Filtering
    L_Filter = I
    H_Filter = I
    for i in range(kk):
        L_Filter = I.dot(I - 0.5 * Ls)
        H_Filter = I.dot(0.5 * Ls)
    F_ = L_Filter + lambda_ * H_Filter
    X_bar = F_.dot(X)
    XXt_bar = X_bar.dot(X_bar.T)
    num_data = X_bar.shape[0]

    #Contrastive nbrs
    nbr = np.zeros((X_bar.shape[0], 10))
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto').fit(X_bar)
    dis, idx = nbrs.kneighbors(X_bar)
    for i in range(X.shape[0]):
        for j in range(10):
                nbr[i][j] += idx[i][j + 1]
    idx = nbr
    idx = idx.astype(np.int)
    id = np.array(idx)
    tmp = np.linalg.inv(1000 * I + XXt_bar)
    S_re = tmp.dot(1000 * Ls + XXt_bar)

    # break point
    trigger = 0
    cf = None
    loss_last = 1e16
    ac_results = 0
    nmi_results = 0
    f1_results = 0
    for m in range(num_data):
        S_re[m][m] = 0
    for epoch in range(30):
        if trigger >= 20:
            break
        grad = np.zeros((num_data, num_data))
        X_Xt_S = XXt_bar.dot(S_re)

        # grad
        for i in range(num_data):
            k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
            for j in range(i+1,num_data):
                F11 = XXt_bar[i][j]
                F12 = X_Xt_S[i][j]
                if j in id[i]:
                    F2 = -1 + 10 * np.exp(S_re[i][j]) / k0
                else:
                    F2 = 10 * np.exp(S_re[i][j]) / k0
                F1 = -2 * F11 + 2 * F12
                grad[i][j] = a * F2 + F1
                grad[j][i] = grad[i][j]

        #Loss
        loss_all_node = 0
        loss_view = 0
        for i in tqdm((range(num_data))):
            k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
            loss_nbr = 0
            for z in range(10):
                if id[i][z] != i:
                    loss_nbr = loss_nbr - np.log(np.exp(S_re[i][id[i][z]]) / k0)
            loss_all_node = loss_all_node + loss_nbr
        loss_view = loss_view + np.linalg.norm(
            X_bar.T - X_bar.T.dot(S_re)) ** 2
        loss_S_re = a * loss_all_node + loss_view
        if loss_S_re < loss_last:
            loss_last = loss_S_re
        if loss_S_re > loss_last:
            break
        S_re, cf = adam(S_re, grad, cf)
        C = 0.5 * (np.fabs(S_re) + np.fabs(S_re.T))
        u, s, v = sp.linalg.svds(C, k=k, which='LM')

        #Clustering
        kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
        predict_labels = kmeans.predict(u)

        re_ = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = re_.evaluationClusterModelFromLabel()
        if ac > ac_results:
            ac_results = ac
            nmi_results = nm
            f1_results = f1
        else:
            trigger += 1
    print("ac = {:>.6f}".format(ac_results),
              "nmi = {:>.6f}".format(nmi_results),
              "f1 = {:>.6f}".format(f1_results))

if __name__ == "__main__":
    dataset = 'cora'
    data = sio.loadmat('./data/{}.mat'.format(dataset))
    Run(data)