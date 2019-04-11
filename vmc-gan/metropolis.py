import numpy as np 
import matplotlib.pyplot as plt

def metropolis(f, nwalkers, nstep, eps = 3, boundary = 2,**kwargs):

	''' perform a MC sampling of the function f
	Args:
		f (func) : function to sample
		nstep (int) : number of mc step
		nwalkers (int) : number of walkers
		eps (float) : size of the mc step
		boudnary (float) : boudnary of the space
		kwargs : argument for f 

	Returns:
		X (list) : position of the walkers
	'''

	X = boundary * (-1 + 2*np.random.rand(nwalkers))
	fx = f(X,**kwargs)
	ones = np.ones(nwalkers)	

	for istep in range(nstep):

		# new position
		xn =  X + eps * (2*np.random.rand(nwalkers) - 1)	

		# new function
		fxn = f(xn,**kwargs)
		df = fxn/fx

		# probability
		P = np.minimum(ones,df)
		tau = np.random.rand(nwalkers)

		# update
		index = P-tau>=0
		X[index] = xn[index]
		fx[index] = fxn[index]
	
	return X


def psi(x,**kwargs):
	''' Compute the value of the wave function.

	Args:
		x: position of the electron
		kwargs: argument of the wf

	Returns: values of psi
	'''
	beta = kwargs['beta']
	norm = np.sqrt(np.pi/beta)
	return np.exp(-beta*x**2)

def rho(x,**kwargs):
	return psi(x,**kwargs)**2

def kinetic(psi,x,eps=1E-6,**kwargs):
	return 1./eps/eps * ( psi(x+eps,**kwargs) - 2*psi(x,**kwargs) + psi(x-eps,**kwargs) )

def potential(x):
	return 0.5*x**2

def local_energy(psi,x,**kwargs):
	return - 0.5 * kinetic(psi,x,**kwargs)/psi(x,**kwargs) + potential(x)

def total_energy(psi,x,**kwargs):
	return 1./len(X) * np.sum(local_energy(psi,x,**kwargs))


beta = 0.5
X = metropolis(rho,1000,1000,beta=beta)
print(total_energy(psi,X,beta=beta))
plt.hist(X,density=True)

xs = np.linspace(-2,2,100)
fxs = rho(xs,beta=beta)
plt.plot(xs,fxs/np.linalg.norm(fxs))
plt.show()


