from implementations import *
from project_helpers import *
import csv
import numpy as np

#Load train data
y, tx, ids = load_csv_data('Data/train.csv', sub_sample=False)
tx[tx == -999] = 0

#Load test data
y_t, tx_t, ids_t = load_csv_data('Data/test.csv', sub_sample=False)
tx_t[tx_t == -999] = 0

#Define parameters
degrees = range(1,10)
lambdas = np.logspace(-15,0)

#Basic linear regression
tr,te = model_pick_ridge(tx, y)
mse,deg,lamb = te[0]

#Linear regression with polynomial basis expansion
tr,te = model_pick_ridge(tx, y, degrees=degrees)
mse,deg,lamb = te[0]

#Ridge regression with polynomial basis expansion
tr,te = model_pick_ridge(tx, y, degrees=range(deg, deg+1), lambdas=lambdas)
mse,deg,lamb = te[0]

tx_deg = build_poly(tx, deg)
w = ridge_regression(y, tx_deg, lamb)[0]

tx_t = build_poly(tx_t, deg)
pred = predict_labels(w, tx_t)

create_csv_submission(ids_t, pred, 'pred_fullmodel_polyexpansion_ridgereg')
