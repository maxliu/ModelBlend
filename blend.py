#!/usr/bin/env/ python
"""
Author : xinyulrsm@gmail.com

Date   : Sat 17:15:53 12/19/2015
"""
import numpy as np
from sklearn.cross_validation import StratifiedKFold


class modelBlent():
    """ for classification only #TODO shoud condsider other forms of output

    Parameters
    ----------


    Attributes
    ----------

    Notes
    -----

    See Also
    --------

    """

    def __init__(self, clfs, blentClf, n_folds=3):

        self.clfs = clfs
        self.blentClf = blentClf
        self.n_folds = n_folds

    def fit(self, X, y):

        # prepare data for blentClf
        y_hat_all = None
        skf = StratifiedKFold(y, n_folds=self.n_folds, shuffle=True)

        for i, (train_index, test_index) in enumerate(skf):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for clf in self.clfs:
                clf.fit(X_train, y_train)

            # last column is test target not prediction
            y_pre = self.clfs_predict(X_test)

            y_hat = np.append(y_pre,
                              np.array([y_test]).transpose(), axis=1)

            y_hat_all = y_hat if i == 0 else np.append(y_hat_all, y_hat, axis=0)

        # fit clfs and blentClf

        for clf in self.clfs:
            clf.fit(X, y)

        self.blend_X = y_hat_all[:, :-1]
        self.blend_y = y_hat_all[:, -1]  # last column is y_test

        self.blentClf.fit(self.blend_X, self.blend_y)

    def predict(self, X):
        # a matrix with row : samples,  column : clfs
        y_hat = self.clfs_predict(X)

        y_out = self.blentClf.predict_proba(y_hat)[:, 1]
        y_pre = self.blentClf.predict(y_hat)

        return y_out, y_pre

    def clfs_predict(self, X):
        """

        Parameters
        ----------


        Return
        ------
        Arrray/Matrix of predictions from clfs

        Note
        ----

        Example
        -------
        prediction from clf_1, [0.1, 0.2, 0.3. 0.4]
        prediction from clf_2, [0.5, 0.6, 0.7. 0.8]
        return a np array like below:
            [
            [0.1, 0.5],
            [0.2, 0.6],
            [0.3, 0.7],
            [0.4, 0.8]
            ]


        >>> clf_predict(self, clf, X_test)

        """
        for i, clf in enumerate(self.clfs):

            if not hasattr(clf, "predict_proba"):
                print "%s deosn't have predict_proba method" % (clf.__class__.__name__)
                raise TypeError

            ypre = np.array([clf.predict_proba(X)[:,1]]).transpose()
            #print "in predict, ypre", ypre

            if i == 0:
                y = ypre
            else:
                y = np.append(y, ypre, axis=1)
            #print "in predict, y", y
        return y


