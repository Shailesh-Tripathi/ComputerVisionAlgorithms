import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

## Q1

# Function to compute mean. Inputs:( image, xOrder(column), yOrder(row))
def computeSpatialMoment(img, p, q):
	moment = 0.0
	for r in range(0, len(img)):
		for c in range(0, len(img[0])):
			moment += (pow(r, q) * pow(c, p) * img[r][c])

	return moment

# Function to compute similitude moments. Inputs:(image, xOrder(column), yOrder(row))
def computeSimilitudeMoment(img, i, j):
	moment = 0.0
	
	# Zeroth order moments
	m00 = computeSpatialMoment(boxIm, 0, 0)
	
	#First order moments
	m10 = computeSpatialMoment(boxIm, 1, 0)
	m01 = computeSpatialMoment(boxIm, 0, 1)

	centroidC = m10 / m00
	centroidR = m01 / m00

	for r in range(0, len(img)):
		for c in range(0, len(img[0])):
			moment += (pow(r - centroidR, j) * pow(c - centroidC, i) * img[r][c])

	moment = moment / pow(m00, 0.5 * (i+j) + 1)
	return moment



def similitudeMoments(boxIm):
	# Similitude Moments
	N=[]
	# Use the condition 2<= i+j <= 3 to compute values of i and j
	for i in range(0,4):
		jLower = max(0, 2-i)
		jUpper = 3 - i
		for j in range(jLower, jUpper+1):
			N.append(computeSimilitudeMoment(boxIm, i, j))
	return N

numfiles = 4
for filenum in range (0,numfiles):
	filename = 'boxIm' + str(filenum+1) + '.bmp'
	boxIm = io.imread(filename)
	Nvals = similitudeMoments(boxIm)
	print(Nvals)


## Q2
# Load the data
X = np.loadtxt('eigdata.txt')
plt.subplot(2,1,1)
plt.plot(X[:,0], X[:,1],'b.')
plt.axis('equal')

# mean subtract data
m = np.mean(X, axis=0)
Y = X - (np.ones((len(X),1))*m)
plt.subplot(2,1,2)
plt.plot(Y[:,0], Y[:,1], 'r.')
plt.axis('equal')
plt.show()

##Q3

N = len(Y)

# Compute covariance matrix
K = (1.0/(N-1)) * np.matmul(np.transpose(Y), Y)
fig, ax = plt.subplots()
plt.plot(Y[:,0], Y[:,1], 'r.')
plt.axis('equal')

# Compute eigenvalues(V) and eigenvectors(U) from covariance matrix
[V,U] = np.linalg.eig(K)
print(K)
print(V)
print(U)
C = 9.0
for i in range(0,len(V)):
	# Compute half length
	halfLength = np.sqrt(V[i] * C)
	# the columns of U are the axes
	plt.plot([-halfLength*U[0,i],halfLength*U[0,i]], [-halfLength*U[1,i],halfLength*U[1,i]],'g', lw=1)

# plot ellipse
ang=np.arctan(U[1,0]/U[0,0]) / np.pi *180
ax.add_patch(Ellipse(xy=(0,0), width=2*np.sqrt(V[0] * C),height=2*np.sqrt(V[1] * C), angle=ang, fc='none', ec='b'))
plt.show()

## Q4
#Rotate Y using eigenvectors

# Projection of Y on eigenvectors
YRotated = np.matmul(Y, np.transpose(U))

# plot results
plt.plot(YRotated[:,0], YRotated[:,1], 'r.')
plt.axis('equal')
plt.show()

# plot histogram after projecting to y-axis
nbins = int(max(YRotated[:,1]) - min(YRotated[:,1]))
plt.hist(YRotated[:,1], nbins,orientation=u'horizontal')
plt.show()
