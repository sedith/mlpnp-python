### IMPORTS
import numpy as np
import sympy as sp
from math import sqrt
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
    R = sp.simplify((sp.eye(3) * (1-sp.cos(phi))*N*N + sp.sin(phi)*N).subs([(phi_,phi)]))
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
    # sp.pprint(F)
    J = sp.simplify(F.jacobian([r1,r2,r3,t1,t2,t3]))
    # sp.pprint(J)

    # sp.pprint(J[0,0])
    # sp.pprint(J[0,1])
    # sp.pprint(J[0,2])
    # sp.pprint(J[0,3])
    # sp.pprint(J[0,4])
    # sp.pprint(J[0,5])
    # print()
    # sp.pprint(J[1,0])
    # sp.pprint(J[1,1])
    # sp.pprint(J[1,2])
    # sp.pprint(J[1,3])
    # sp.pprint(J[1,4])
    # sp.pprint(J[1,5])
    # phi__ = phi_.subs([(r1,-0.30434332) , (r2,-0.250293) , (r3,-0.27863592)])
    # print(sqrt((-0.30434332)**2 + (-0.250293)**2 + (-0.27863592)**2))
    # J = J.subs([ (pt1,-0.69754412) , (pt2,-6.09545832), (pt3,-0.58417732) ,
    #                   (nr1,-0.40795141) , (nr2,0.84983699) , (nr3,-0.33369558) ,
    #                   (ns1,-0.90655869) , (ns2,-0.33369558) , (ns3,0.25845428) ,
    #                   (r1,-0.30434332) , (r2,-0.250293) , (r3,-0.27863592) ,
    #                   (t1,2.8757762) , (t2,7.43012996) , (t3,3.19081123) ,
    #                   (phi,phi__)
    #                 ])
    jac_np = np.matrix([[J[0],J[1],J[2],J[3],J[4],J[5]], \
                        [J[6],J[7],J[8],J[9],J[10],J[11]]])
    #
    #   [ 3.00203466  0.38069314 -0.96498182 -0.40795141  0.84983699 -0.33369558]
    #   [-1.44983394 -0.09749644 -2.6130425  -0.90655869 -0.33369558  0.25845428]
    #
    # print()
    print('sympy    :\n',jac_np)
    #
    # print() ; print()
