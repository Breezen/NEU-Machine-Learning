import time
import scipy.io as sio
import regression as reg

n = input("Enter polynomial degree n: ")
batch_size = input("Enter the SGD batch_size: ")

data = sio.loadmat("../HW1_Data/dataset1.mat")
x_train, x_test = data["X_trn"], data["X_tst"]
y_train, y_test = data["Y_trn"], data["Y_tst"]

X_train = reg.non_linear_trans(x_train, n)
X_test = reg.non_linear_trans(x_test, n)

start = time.time()
theta_closed_form = reg.vanilla_closed_form(X_train, y_train)
end = time.time()
print "\nClosed-form: (time elapsed: %fs)" % (end - start)
print "theta:\n", theta_closed_form
print "training error: ", reg.get_error(theta_closed_form, X_train, y_train)
print "test error: ", reg.get_error(theta_closed_form, X_test, y_test)

start = time.time()
theta_SGD = reg.vanilla_SGD(X_train, y_train, batch_size=batch_size)
end = time.time()
print "\nSGD: (time elapsed: %fs)" % (end - start)
print "theta:\n", theta_SGD
print "training error: ", reg.get_error(theta_SGD, X_train, y_train)
print "test error: ", reg.get_error(theta_SGD, X_test, y_test)