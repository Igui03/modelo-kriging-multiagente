import numpy as np
import matplotlib.pyplot as plt

#Covarianza
def rbf_kernel(x1,x2,len_scale=1.0):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sqdist = np.sum((x1[:,np.newaxis] - x2[np.newaxis,:])**2,axis=2)
    return np.exp(-0.5/len_scale**2*sqdist)

#Puntos donde se evalúa el proceso gaussiano
X = np.linspace(-5,5,100).reshape(-1,1)

#Distribución prior
K_prior = rbf_kernel(X,X)
mu_prior = np.zeros(X.shape[0])
samples_prior = np.random.multivariate_normal(mean=mu_prior,cov=K_prior,size=5)

#Medidas
medidas_x = np.array([[-3.2],[-1.0],[2.5]])
medidas_y = np.array([-1.2,2.0,-0.5])

#Matrices de covarianza
K_xx = rbf_kernel(medidas_x,medidas_x)+1e-8*np.eye(len(medidas_x))  # Estabilidad numérica
K_xs = rbf_kernel(medidas_x,X)
K_ss = rbf_kernel(X,X)

#Condicionamiento bayesiano
K_xx_inv = np.linalg.inv(K_xx)
mu_post = K_xs.T@K_xx_inv@medidas_y
cov_post = K_ss-K_xs.T@K_xx_inv@ K_xs

#Evaluación de la distribución posterior
samples_post = np.random.multivariate_normal(mu_post,cov_post,size=5)

#Representación
fig, axs = plt.subplots(1, 2, figsize=(22, 7))

#Distribución prior
for i in range(5):
    axs[0].plot(X,samples_prior[i],lw=1)
std_prior = np.sqrt(np.diag(K_prior))
axs[0].fill_between(X.ravel(),mu_prior-2*std_prior,mu_prior+2*std_prior,color='lightgray',alpha=0.5,label='Intervalo 2σ')
axs[0].plot(X,mu_prior,'--k',lw=2,label="Media prior")
axs[0].set_title("Distribución prior",fontsize=22)
axs[0].legend(fontsize='x-large',loc='upper left')
axs[0].set_ylim(-3, 3)
axs[0].set_xlabel('x (-)')
axs[0].set_ylabel('y (-)')
axs[0].grid(True)

#Distribución posterior
for i in range(5):
    axs[1].plot(X,samples_post[i],lw=1)
std_post = np.sqrt(np.diag(cov_post))
axs[1].fill_between(X.ravel(),mu_post-2*std_post,mu_post+2*std_post,color='lightgray',alpha=0.5,label='Intervalo 2σ')
axs[1].plot(X,mu_post,'--k',lw=2,label="Media posterior")
axs[1].scatter(medidas_x,medidas_y,color='red',marker='x',lw=3,label="Datos observados")
axs[1].set_title("Distribución posterior",fontsize=22)
axs[1].set_ylim(-3, 3)
axs[1].legend(fontsize='x-large',loc='upper left')
axs[1].set_xlabel('x (-)')
axs[1].set_ylabel('y (-)')
axs[1].grid(True)

plt.tight_layout()
plt.show()
