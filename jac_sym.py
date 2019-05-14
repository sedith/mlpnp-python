### IMPORTS
import numpy as np
import sympy as sp

### MAIN
if __name__ == '__main__':


    sp.init_printing(use_unicode=True)


    pt1,pt2,pt3 = sp.symbols('pt1,pt2,pt3', real=True)

    nr1,nr2,nr3 = sp.symbols('nr1,nr2,nr3', real=True)
    ns1,ns2,ns3 = sp.symbols('ns1,ns2,ns3', real=True)

    r1,r2,r3,t1,t2,t3 = sp.symbols('r1,r2,r3,t1,t2,t3', real=True)
    phi = sp.symbols('phi')
    rod = sp.Matrix([r1,r2,r3])
    phi_ = rod.norm()

    N = sp.Matrix([[0,-r3,r2],[r3,0,-r1],[-r2,r1,0]])
    R = sp.simplify((sp.eye(3) * (1-sp.cos(phi_))*N*N + sp.sin(phi_)*N).subs([(phi_,phi)]))
    # sp.pprint(R)

    R11 = R[0,0] ; R12 = R[0,1] ; R13 = R[0,2]
    R21 = R[1,2] ; R22 = R[1,2] ; R23 = R[1,2]
    R31 = R[2,2] ; R32 = R[2,2] ; R33 = R[2,2]

    f1 =  nr1*(R11*pt1+R12*pt2+R13*pt3 + t1) \
        + nr2*(R21*pt1+R22*pt2+R23*pt3 + t2) \
        + nr3*(R31*pt1+R32*pt2+R33*pt3 + t3)
    f2 =  ns1*(R11*pt1+R12*pt2+R13*pt3 + t1) \
        + ns2*(R21*pt1+R22*pt2+R23*pt3 + t2) \
        + ns3*(R31*pt1+R32*pt2+R33*pt3 + t3)

    F = sp.Matrix([sp.simplify(f1),sp.simplify(f2)])
    # sp.pprint(sp.simplify(F))
    sp.pprint(sp.simplify(F.jacobian([r1,r2,r3,t1,t2,t3])))
    # jac_sym = F.jacobian([r1,r2,r3,t1,t2,t3]).subs([(r1,rot[0]),(r2,rot[1]),(r3,rot[2]),(t1,trans[0]),(t2,trans[1]),(t3,trans[2])])
    # jac_np = np.matrix([[jac_sym[0],jac_sym[1],jac_sym[2],jac_sym[3],jac_sym[4],jac_sym[5]], \
    #                     [jac_sym[6],jac_sym[7],jac_sym[8],jac_sym[9],jac_sym[10],jac_sym[11]]])
    # print()
    # print('sympy    :\n',jac_np)
    #
    # print() ; print()
