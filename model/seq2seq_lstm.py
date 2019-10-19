from sklearn import preprocessing
from utils.model import binary_matrix


def prepare_data(train, l, r = 0.8, h = 1, p = 0.6):
    # get specific numbers of time-series
    rows = train.shape[0]
    cols = train.shape[1]
    T = cols * p
    train = train[:, 0:T]

	X, Y = list(), list()
    train = train.transpose()
	for in_start in range(0, len(train) - l - h):        
        # Data normalization
        train = preprocessing.normalize(train)

        # Data need to be transformed as format: (x,y) where x is the input with shape (K,l), y is the target with shape (K,h)
        x_input = train[:, in_start:l]
        X.append(X)
        y_input = train[:, l:(l+h)]
        Y.append(Y)
        
        # Randomly create the binary matrix M_(K×l) which (∑(M) )/(K×l)=r
        binary_matrix(r, rows, cols)

        # Change the value x_i^k whose m_i^k=0 as x_i^k←random(x_i^k-x ´,x_i^k+x ´ ), where x ´ is the stdev of the training set

        # Add to (x, y) to data_loader

	return array(X), array(Y)