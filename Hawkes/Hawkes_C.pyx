import numpy as np
from scipy.special import gamma,digamma
cimport numpy as np

def LG_kernel_SUM_exp_i_cython(np.ndarray[np.float64_t,ndim=1] T, double alpha, double beta):

    cdef int n = T.shape[0]

    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_a = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_b = np.zeros(n, dtype=np.float64)
    
    cdef np.ndarray[np.float64_t,ndim=1] dt = T[1:] - T[:-1]
    cdef np.ndarray[np.float64_t,ndim=1] r = np.exp(-beta*dt)

    cdef double x = 0.0
    cdef double x_a = 0.0
    cdef double x_b = 0.0

    cdef int i;

    for i in xrange(n-1):
        x   = ( x   + alpha*beta  ) * r[i]
        x_a = ( x_a +       beta  ) * r[i]
        x_b = ( x_b + alpha       ) * r[i] - x*dt[i]
        
        l[i+1] = x
        dl_a[i+1] = x_a
        dl_b[i+1] = x_b

    return [l,dl_a,dl_b]


def LG_kernel_SUM_pow_cython(np.ndarray[np.float64_t,ndim=1] T,double k, double p, double c):
    
    cdef int n = T.shape[0]
    
    cdef np.ndarray[np.float64_t,ndim=1] l    = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_p = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_k = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] dl_c = np.zeros(n, dtype=np.float64)
    
    cdef int num_div = 16
    cdef double delta = 1.0/num_div
    cdef np.ndarray[np.float64_t,ndim=1] s = np.linspace(-9,9,num_div*18+1)
    cdef np.ndarray[np.float64_t,ndim=1] log_phi = s-np.exp(-s)
    cdef np.ndarray[np.float64_t,ndim=1] log_dphi = log_phi + np.log(1+np.exp(-s))
    cdef np.ndarray[np.float64_t,ndim=1] phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))
    cdef np.ndarray[np.float64_t,ndim=1] dphi = np.exp(log_dphi) # dphi = phi*(1+np.exp(-s))
    
    cdef np.ndarray[np.float64_t,ndim=1] H      = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
    cdef np.ndarray[np.float64_t,ndim=1] H_p = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p) * (log_phi-digamma(p))
    cdef np.ndarray[np.float64_t,ndim=1] H_c = delta * k * np.exp( log_dphi +     p*log_phi - c*phi ) / gamma(p) * (-1)
    
    cdef np.ndarray[np.float64_t,ndim=1] g = np.zeros_like(s)
    
    cdef int i
    
    for i in range(n-1):
        g = (g+1)*np.exp( - phi*(T[i+1]-T[i]) )
        l[i+1] = g.dot(H)
        dl_k[i+1] = l[i+1]/k
        dl_p[i+1] = g.dot(H_p)
        dl_c[i+1] = g.dot(H_c)
    
    return [l,dl_k,dl_p,dl_c]

