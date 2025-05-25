#Kriging universal con UBC

import numpy as np
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
import gstools as gs
from scipy.optimize import direct
from scipy.integrate import solve_ivp

s = 12
np.random.seed(s)
L = 50
bounds = ((0,L),(0,L))
lim = [0,L,0,L]
grid_size = 51
X = np.linspace(0,L,grid_size)
Y = np.linspace(0,L,grid_size)


#Campo aletorio de GSTools
model = gs.Gaussian(dim=2,var=15,len_scale=15)
srf = gs.SRF(model,seed=s)
campo_real = srf((X,Y),mesh_type='structured')

#Puntos iniciales
n_agentes = 6
puntos_x = [24,25,26,26,25,24]
puntos_y = [26,26,26,24,24,24]
puntos_z = [srf([puntos_x[i],puntos_y[i]],seed=s)[0] for i in range(n_agentes)] 
S = np.array([[puntos_x[i],puntos_y[i],puntos_z[i]] for i in range(n_agentes)]) #Conjunto de medidas inicial

#Mejor estimación del campo
phi_max = max(puntos_z)
coord = np.argmax(puntos_z)
pos_max = np.array([S[coord,0],S[coord,1]])

#Función J restringida a D
def Jres(x,xi,xd,uk,alpha,beta,b,phi_max):
    
    #Término repulsivo a los puntos objetivo xd
    j = 0
    for i in range(len(xd)):
        j += (xd[i]-x)@(xd[i]-x)

    #Condición UBC
    phi,var = uk.execute('points',x[0],x[1])
    cond = phi+b*(var**0.5)-phi_max
    
    #Polo a la cercanía de las medidas
    medidas = np.array([s[:2] for s in S[-n_agentes:]])
    dist = min([np.linalg.norm(medidas[i]-x) for i in range(len(pos))])

    return alpha*(x-xi)@(x-xi)-beta*j+1e6*(cond<=0)+1e6*(dist<=5e-1)

#Función para buscar los puntos objetivo de los agentes
def busca_puntos(pos,uk,alpha,beta,b,phi_max):
    xd = np.empty((0,2))
    #Primero busca para agente 0, luego para el agente 1 con repulsión de xd[0], etc.
    for i in range(len(pos)):
        xi = pos[i]
        new_pos = direct(Jres,bounds,args=(xi,xd,uk,alpha,beta,b,phi_max),maxfun=2000,len_tol=1e-3)
        xd = np.vstack([xd,[new_pos.x]])
    return xd

b = 1
alpha = 1e-2
beta = 1/n_agentes

#Estimación inicial del campo
uk = UniversalKriging(
    puntos_x,puntos_y,puntos_z,
    variogram_model='gaussian',
    drift_terms=["regional_linear"]
)
campo_estimado,varianza_estimada = uk.execute("grid",X,Y)
#Para pintar la zona donde cond<0
D = np.ma.masked_where(~(campo_estimado+b*(varianza_estimada**0.5)-phi_max<0),campo_estimado)

#Ley de control
M = 1.0
k1 = 1
k2 = 2
k3 = 100
q = 0.1
paso = 0.01
t0 = 0
t_max = 100000
t_rango = (t0,t_max)
t_eval = np.arange(t0,t_max,paso)
eps = 0.01

def control(t,y,xd,M,n,k1,k2,k3,q,eps):
    pos = y[:2*n].reshape(n,2)
    vel = y[2*n:].reshape(n,2) 
    vd = np.zeros((n,2))
    acc = np.zeros((n,2))
    for i in range(n):
        xi = pos[i]
        vi = vel[i]
        xdi = xd[i]
        vdi = vd[i]
        cercania = xi-xdi
        frenado = vi-vdi
        repulsion = 0
        for j in range(n):
            if j != i:
                xj = pos[j]
                repulsion += (xi-xj)*np.exp(-(xi-xj)@(xi-xj)/q)/q
        u = -k3*cercania-k1*frenado+2*k2*repulsion
        acel = u/M
        #Velocidad máxima v_max
        v_max = 5
        mod = np.linalg.norm(acel)
        acc[i] = acel       
        if np.linalg.norm(vel[i])>v_max:
            vel[i] = v_max*vel[i]/np.linalg.norm(vel[i])
    return np.concatenate([vel.flatten(),acc.flatten()])

#Condición de parada de la integración
def rango(t,y,xd,M,n,k1,k2,k3,q,eps):
    pos = y[:2*n].reshape(n,2)
    dists = [np.linalg.norm(pos[i]-xd[i]) for i in range(n)]
    r = min(dists)-eps
    return r
rango.terminal = True   #Para que salte en ese evento
rango.direction = -1   #Salta cuando hay cambio de signo

#Inicio de la búsqueda
it = 0
#Estado inicial
pos = np.array([s[:2] for s in S[-n_agentes:]])
v0 = np.zeros((n_agentes,2))
y0 = np.concatenate([pos.flatten(),v0.flatten()])
xd = busca_puntos(pos,uk,alpha,beta,b,phi_max)

#Representación de la estimación inicial del campo
lw = 2    #Para pintar los puntos más grande
fig,ax = plt.subplots(1,1)
im = ax.contourf(X,Y,campo_real.T,100,extent=lim,origin="lower",cmap='copper')
ax.scatter(puntos_x,puntos_y,color="lightgreen",marker='o',lw=lw,label="Agentes")
[ax.scatter(xd[i][0],xd[i][1],marker='*',color='blue',label='Objetivos') for i in range(len(xd))]
plt.xlabel('x (-)')
plt.ylabel('y (-)')
plt.colorbar(im)
handles,labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys(),fontsize='small')
ax.set_title('Campo real',fontsize=15)
plt.show()

cond = campo_estimado+b*(varianza_estimada**0.5)-phi_max
while (np.max(cond)>0) & (it<200):
    print(f'it={it} , D_points={np.sum(cond>0)} , phi_max={phi_max} , t={t0}')
    xd = busca_puntos(pos,uk,alpha,beta,b,phi_max)
    
    if it%5==0:
        fig,axes = plt.subplots(1,2,figsize=(22,7))
        
        ax = axes[0]
        im = ax.contourf(X,Y,campo_real.T,100,extent=lim,origin="lower",cmap='copper')
        ax.scatter(pos[:,0],pos[:,1],color="lightgreen",marker='o',lw=lw,label="Agentes")
        [ax.scatter(xd[i][0],xd[i][1],marker='*',color='blue',label='Objetivos') for i in range(len(xd))]
        ax.scatter(S[:,0],S[:,1],color="white",marker='.',lw=lw,label="Medidas")
        ax.contourf(X,Y,D.T,levels=1,hatches=['////'],colors='none',extent=lim,origin='lower') #Para pintar la zona prohibida por la restricción
        ax.set_title('Campo real tras %r medidas'%(it+n_agentes),fontsize=15)
        plt.xlabel('x (-)')
        plt.ylabel('y (-)')
        plt.colorbar(im)
        ax.set_aspect('equal')
        handles,labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels,handles))
        ax.legend(by_label.values(),by_label.keys(),fontsize='small')   

        ax = axes[1]
        im = ax.contourf(X,Y,campo_estimado,100,extent=lim,origin="lower",cmap='copper')
        ax.scatter(pos[:,0],pos[:,1],color="lightgreen",marker='o',lw=lw,label="Agentes")
        [ax.scatter(xd[i][0],xd[i][1],marker='*',color='blue',label='Objetivos') for i in range(len(xd))]
        ax.scatter(S[:,0],S[:,1],color="white",marker='.',lw=lw,label="Medidas")
        ax.set_title('Campo estimado tras %r medidas'%(it+n_agentes),fontsize=15)
        plt.xlabel('x (-)')
        plt.ylabel('y (-)')
        plt.colorbar(im)
        ax.set_aspect('equal')
        handles,labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels,handles))
        ax.legend(by_label.values(),by_label.keys(),fontsize='small')
        plt.show()  
            
    y = solve_ivp(control,t_rango,y0,t_eval=t_eval,events=rango,args=(xd,M,n_agentes,k1,k2,k3,q,eps),max_step=0.005,dense_output=True)
    print('y.status=',y.status)   #Si y.status=1 ha saltado el evento; si y.status=-1, no 
    t0 += y.t_events[0][0]   #Se va actualizando el tiempo inicial de la evolución temporal
    t_rango = (t0,t_max)
    t_eval = np.arange(t0,t_max,paso)
    
    y0 = y.sol(y.t_events[0][0])   #El estado final es el estado inicial de la siguiente búsqueda
    pos = y0[:2*n_agentes].reshape(-1,2)
    ind = np.argmin([np.linalg.norm(pos[i]-xd[i]) for i in range(len(pos))])   #Agente que ha llegado al objetivo
    print(f'El agente {ind} en {pos[ind]} ha llegado a {xd[ind]} (dist = {np.linalg.norm(pos[ind]-xd[ind])})')
    nueva_z = srf([pos[ind][0],pos[ind][1]],seed=s)[0]
    nuevo_punto = [pos[ind][0],pos[ind][1],nueva_z]
    S = np.append(S,np.array([nuevo_punto]),0)
    if nueva_z>phi_max:
        phi_max = nueva_z.copy()
        pos_max = np.array([pos[ind][0],pos[ind][1]])
    uk = UniversalKriging([s[0] for s in S],[s[1] for s in S],[s[2] for s in S],variogram_model="gaussian",drift_terms=["regional_linear"])
    campo_estimado,varianza_estimada = uk.execute("grid",X,Y)
    D = np.ma.masked_where(~(campo_estimado+b*(varianza_estimada**0.5)-phi_max<0),campo_estimado)
    cond = campo_estimado+b*(varianza_estimada**0.5)-phi_max
    it+=1   

#Representación del final del algoritmo
fig,axes = plt.subplots(1,2,figsize=(22,7))
        
ax = axes[0]
im = ax.contourf(X,Y,campo_real.T,100,extent=lim,origin="lower",cmap='copper')
ax.scatter(S[:n_agentes,0],S[:n_agentes,1],color="lightgreen",marker='o',lw=lw,label="Agentes")
ax.set_title('Campo real (posiciones iniciales)',fontsize=15)
plt.xlabel('x (-)')
plt.ylabel('y (-)')
plt.colorbar(im)
ax.set_aspect('equal')
handles,labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys(),fontsize='small')   

ax = axes[1]
im = ax.contourf(X,Y,campo_estimado,100,extent=lim,origin="lower",cmap='copper')
ax.scatter(pos[:,0],pos[:,1],color="lightgreen",marker='o',lw=lw,label="Agentes")
[ax.scatter(xd[i][0],xd[i][1],marker='*',color='blue',label='Objetivos') for i in range(len(xd))]
ax.scatter(S[:,0],S[:,1],color="white",marker='.',lw=lw,label="Medidas")
ax.set_title('Campo estimado tras %r medidas'%(it+n_agentes),fontsize=15)
plt.xlabel('x (-)')
plt.ylabel('y (-)')
plt.colorbar(im)
ax.set_aspect('equal')
handles,labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys(),fontsize='small')
plt.show()  

#Análisis de resultados
campo_real_max = np.max(campo_real)
i,j = np.unravel_index(np.argmax(campo_real),campo_real.shape)
print(f'Campo real: El valor máximo del campo es {campo_real_max} y se encuentra en {[X[i],Y[j]]}')
print(f'Kriging: La estimación del máximo del campo es {phi_max} y se encuentra en {[pos_max[0],pos_max[1]]}')
err_campo = abs(phi_max-campo_real_max)*100/campo_real_max
err_pos = (pos_max-np.array([X[i],Y[j]]))@(pos_max-np.array([X[i],Y[j]]))*100/(L*(2**0.5))  #Normalizado a diagonal, máx. dist. posible
print(f'El error relativo del valor del campo es del {err_campo}%')
print(f'El error en posiciones es del {err_pos}%')
