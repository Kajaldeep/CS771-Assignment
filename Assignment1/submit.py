import numpy as np
from sklearn.svm import LinearSVC
import sklearn

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	
	R = 64
	S = 4

	A1 = Z_train[:,64:68]
	A2 = Z_train[:,68:72]

	a1 = A1[:,0]*2**3+A1[:,1]*2**2+A1[:,2]*2**1+A1[:,3]*2**0
	a2 = A2[:,0]*2**3+A2[:,1]*2**2+A2[:,2]*2**1+A2[:,3]*2**0

	f1=np.zeros((len(Z_train),(16*(R+1))), dtype=int)
	f2=np.zeros((len(Z_train),(16*(R+1))), dtype=int)

	for i in range(len(Z_train)):
		f1[i,int(a1[i]*(R+1)):int((a1[i]+1)*(R+1))]=np.append(Z_train[i,0:64],[1])
		f2[i,int(a2[i]*(R+1)):int((a2[i]+1)*(R+1))]=np.append(Z_train[i,0:64],[1])

	f=np.subtract(f1,f2)
	clf = LinearSVC( loss = "squared_hinge" )
	clf.fit( f, Z_train[ :, -1 ] )
	return clf


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	
	R = 64
	S = 4

	A1 = X_tst[:,64:68]
	A2 = X_tst[:,68:72]

	a1 = A1[:,0]*2**3+A1[:,1]*2**2+A1[:,2]*2**1+A1[:,3]*2**0
	a2 = A2[:,0]*2**3+A2[:,1]*2**2+A2[:,2]*2**1+A2[:,3]*2**0

	f1=np.zeros((len(X_tst),(16*(R+1))), dtype=int)
	f2=np.zeros((len(X_tst),(16*(R+1))), dtype=int)

	for i in range(len(X_tst)):
		f1[i,int(a1[i]*(R+1)):int((a1[i]+1)*(R+1))]=np.append(X_tst[i,0:64],[1])
		f2[i,int(a2[i]*(R+1)):int((a2[i]+1)*(R+1))]=np.append(X_tst[i,0:64],[1])
	f=np.subtract(f1,f2)
	return model.predict( f )
