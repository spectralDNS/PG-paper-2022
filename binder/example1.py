import sympy as sp
import array_to_latex as a2l
from shenfun import *
from shenfun.jacobi.recursions import half, cn, Lmat
config['assembly']['splitmeasure'] = True

x = sp.Symbol('x', real=True)
a = 1
ue1 = sp.exp(-x**4*sp.Rational(1,4))*sp.integrate(100*sp.exp(-x**4*sp.Rational(1,4))*sp.sin(5*x**2), (x, -1, x))
fe1 = sp.diff(ue1, x, 1) + x**3*ue1
ue2 = sp.exp(-x**4/4)*(x+1)
#ue2 = sp.cos(2*sp.pi*x)*(x+1)
fe2 = (a*x**2+1)*sp.diff(ue2, x, 1) + ue2

olver = {
    1: (ue1, fe1),
    2: (ue2, fe2),
}

def main(N, method=1, family='C', eq=1, returnmat=False):
    SN = FunctionSpace(N, family, bc=(0, None))
    VN = FunctionSpace(N+1, family, basis='Phi1')
    u = TrialFunction(SN)
    v = TestFunction(VN)
    if eq == 1:
        D = inner(u, x**3*v) + inner(grad(u), v)
    elif eq == 2:
        D = [inner(u, v)] + inner((a*x**2+1)*grad(u), v)

    if returnmat:
        return np.sum(np.array(D, dtype=object))

    sol = la.Solver(D)
    u_hat = Function(SN)
    X = SN.mesh()
    ue, fe = olver[eq]
    if eq == 1:
        ul = sp.lambdify(x, ue)
        uq = Array(SN, buffer=np.array([ul(j) for j in X]))
    else:
        uq = Array(SN, buffer=ue)

    if method == 1:
        M = 400
        VM = FunctionSpace(M, family, basis='Phi1')
        fM = Function(VM)
        fM = VM.scalar_product(Array(VM, buffer=fe), fM)
        f_hat = Function(VN)
        f_hat[:N-1] = fM[:N-1]
        u_hat = sol(f_hat, u_hat)
    elif method == 2:
        T = VN.get_orthogonal()
        fN = Array(T, buffer=fe)
        f_hat = inner(v, fN)
        u_hat = sol(f_hat, u_hat)
    elif method == 3: # quasi
        T = SN.get_orthogonal()
        fN = Function(T, buffer=fe).refine(N+1)
        f_hat = inner(v, fN)
        u_hat = sol(f_hat, u_hat)

    uj = u_hat.backward()
    error = np.sqrt(inner(1, (uj-uq)**2))
    return error

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cols = ('k',)
    eq = 2
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for c, fam in zip(cols, 'L'):
        #M = (2**2, 2**3, 2**4, 2**5, 2**6, 2**7)
        error = {}
        M = np.arange(8, 40, 4)
        for method in (1, 2, 3):
            error[method] = []
            for N in M:
                error[method].append(main(N, method, fam, eq))
        ax[0].semilogy(M, error[1], '+', M, error[2], 'o', M, error[3], 's', fillstyle='none', color=c)
    ax[0].legend(['Petrov-Galerkin', '$f \in {P}_{N+1}$', '$f \in {P}_{N}$'])
    ax[0].set(xlabel='N', ylabel='$||u_N-u||$')
    ax[1].spy(main(20, 2, 'C', 2, True).diags(), markersize=5, mec='k', mfc='k', marker='o', aspect='auto')
    ##ax[1].set(xlabel=)
    plt.show()
    ##a2l.to_ltx(np.array([error[1], error[2], error[3]]), frmt='{:6.2e}', print_out=True, mathform=False)
