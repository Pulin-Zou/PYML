from gmm import Normal
from gmm import GMM
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':

    bmu, bstd = 170, 2
    gmu, gstd = 165, 2
    bdata = (bmu + bstd * np.random.randn(4000)).reshape((4000, 1))
    gdata = (gmu + gstd * np.random.randn(2000)).reshape((2000, 1))
    data = np.concatenate((bdata, gdata), axis = 0)
    np.random.shuffle(data)

    gmm = GMM(data, 1, 2)
    priors, params = gmm.em(200)

    fig = plt.Figure()
    n,bins,patches = plt.hist(data, 50, normed= True, facecolor = 'b', alpha = 0.75)

    for i in range(len(priors)):
        y = [priors[i] * params[i].pdf(x)[0][0] for x in bins]
        plt.plot(bins, y, 'r', linewidth=1)
    plt.show()



