### IMPORTS
import numpy as np
import sympy as sp
from math import sqrt

def hisjac():

    nr1,nr2,nr3 = sp.symbols('nr1,nr2,nr3', real=True)
    ns1,ns2,ns3 = sp.symbols('ns1,ns2,ns3', real=True)

    r1 = nr1
    r2 = nr2
    r3 = nr3

    s1 = ns1
    s2 = ns2
    s3 = ns3


    pt1,pt2,pt3 = sp.symbols('pt1,pt2,pt3', real=True)

    X1 = pt1
    Y1 = pt2
    Z1 = pt3

    R1,R2,R3,t1,t2,t3 = sp.symbols('r1,r2,r3,t1,t2,t3', real=True)
    w1 = R1
    w2 = R2
    w3 = R3

    t5 =   w1*w1
    t6 =   w2*w2
    t7 =   w3*w3
    # t8 =   sp.simplify(t5+t6+t7)
    # t9 =   sp.sqrt(t8) # phi
    t9 =   sp.symbols('phi')
    t8 =   t9**2
    t10 =  sp.sin(t9)
    t11 =  1/t9
    t12 =  sp.cos(t9)
    t13 =  t12-1
    t14 =  1/t8
    t16 =  t10*t11*w3
    t17 =  t13*t14*w1*w2
    t19 =  t10*t11*w2
    t20 =  t13*t14*w1*w3

    t24 =  t6+t7
    # t27 =  sp.simplify(-(t16+t17)) # R12
    t27 =  sp.symbols('R12')
    t28 =  Y1*t27
    # t29 =  sp.simplify(t19-t20) # R13
    t29 =  sp.symbols('R13')
    t30 =  Z1*t29
    t31 =  t13*t14*t24
    # t32 =  t31+1 # R11
    t32 =  sp.symbols('R11')
    t33 =  X1*t32
    t15 =  t1+t28+t30+t33
    t21 =  t10*t11*w1
    t22 =  t13*t14*w2*w3
    t45 =  t5+t7
    # t53 =  sp.simplify(t16-t17) # R21
    t53 =  sp.symbols('R21')
    t54 =  X1*t53
    # t55 =  sp.simplify(-(t21+t22)) # R23
    t55 =  sp.symbols('R23')
    t56 =  Z1*t55
    t57 =  t13*t14*t45
    # t58 =  t57+1 # R22
    t58 =  sp.symbols('R22')
    t59 =  Y1*t58
    t18 =  t2+t54+t56+t59
    t34 =  t5+t6
    # t38 =  sp.simplify(-(t19+t20)) # R31
    t38 =  sp.symbols('R31')
    t39 =  X1*t38
    # t40 =  sp.simplify(t21-t22) # R32
    t40 =  sp.symbols('R32')
    t41 =  Y1*t40
    t42 =  t13*t14*t34
    # t43 =  t42+1 # R33
    t43 =  sp.symbols('R33')
    t44 =  Z1*t43
    t23 =  t3+t39+t41+t44
    # t25 =  sp.simplify(1/(t8**(3/2)))
    # t26 =  sp.simplify(1/(t8**2))
    # t35 =  sp.simplify(t12*t14*w1*w2)
    # t36 =  sp.simplify(t5*t10*t25*w3)
    # t37 =  sp.simplify(t5*t13*t26*w3*2)
    # t46 =  sp.simplify(t10*t25*w1*w3)
    # t47 =  sp.simplify(t5*t10*t25*w2)
    # t48 =  sp.simplify(t5*t13*t26*w2*2)
    # t49 =  sp.simplify(t10*t11)
    # t50 =  sp.simplify(t5*t12*t14)
    # t51 =  sp.simplify(t13*t26*w1*w2*w3*2)
    # t52 =  sp.simplify(t10*t25*w1*w2*w3)
    t60 =  t15*t15
    t61 =  t18*t18
    t62 =  t23*t23
    t63 =  t60+t61+t62 # lambda_i **2
    # t64 =  t5*t10*t25
    # t65 =  1/t632 # 1/lambda_i
    t65 = 1/sp.symbols('lambda_i')
    t63 = 1/t65**2

    t66 =  Y1*r2*t6
    t67 =  Z1*r3*t7
    t68 =  r1*t1*t5
    t69 =  r1*t1*t6
    t70 =  r1*t1*t7
    t71 =  r2*t2*t5
    t72 =  r2*t2*t6
    t73 =  r2*t2*t7
    t74 =  r3*t3*t5
    t75 =  r3*t3*t6
    t76 =  r3*t3*t7
    t77 =  X1*r1*t5
    t78 =  X1*r2*w1*w2
    t79 =  X1*r3*w1*w3
    t80 =  Y1*r1*w1*w2
    t81 =  Y1*r3*w2*w3
    t82 =  Z1*r1*w1*w3
    t83 =  Z1*r2*w2*w3
    t84 =  X1*r1*t6*t12
    t85 =  X1*r1*t7*t12
    t86 =  Y1*r2*t5*t12
    t87 =  Y1*r2*t7*t12
    t88 =  Z1*r3*t5*t12
    t89 =  Z1*r3*t6*t12
    t90 =  X1*r2*t9*t10*w3
    t91 =  Y1*r3*t9*t10*w1
    t92 =  Z1*r1*t9*t10*w2
    t102 = X1*r3*t9*t10*w2
    t103 = Y1*r1*t9*t10*w3
    t104 = Z1*r2*t9*t10*w1
    t105 = X1*r2*t12*w1*w2
    t106 = X1*r3*t12*w1*w3
    t107 = Y1*r1*t12*w1*w2
    t108 = Y1*r3*t12*w2*w3
    t109 = Z1*r1*t12*w1*w3
    t110 = Z1*r2*t12*w2*w3
    t93 =  sp.simplify(t66+t67+t68+t69+t70+t71+t72+t73+t74+t75+t76+t77+t78+t79+t80+t81+t82+t83+t84+t85+t86+t87+t88+t89+t90+t91+t92-t102-t103-t104-t105-t106-t107-t108-t109-t110)
    sp.pprint(t93) ; exit()
    # t94 =  sp.simplify(t10*t25*w1*w2)
    # t95 =  sp.simplify(t6*t10*t25*w3)
    # t96 =  sp.simplify(t6*t13*t26*w3*2)
    # t97 =  sp.simplify(t12*t14*w2*w3)
    # t98 =  sp.simplify(t6*t10*t25*w1)
    # t99 =  sp.simplify(t6*t13*t26*w1*2)
    # t100 = sp.simplify(t6*t10*t25)
    t101 = 1/sp.sqrt(t63**3)
    # t111 = sp.simplify(t6*t12*t14)
    # t112 = sp.simplify(t10*t25*w2*w3)
    # print('hello') #4
    # t113 = sp.simplify(t12*t14*w1*w3)
    # t114 = sp.simplify(t7*t10*t25*w2)
    # t115 = sp.simplify(t7*t13*t26*w2*2)
    # t116 = sp.simplify(t7*t10*t25*w1)
    # t117 = sp.simplify(t7*t13*t26*w1*2)
    # t118 = sp.simplify(t7*t12*t14)
    # t119 = sp.simplify(t13*t24*t26*w1*2)
    # t120 = sp.simplify(t10*t24*t25*w1)
    # t121 = sp.simplify(t119+t120)
    # t122 = sp.simplify(t13*t26*t34*w1*2)
    # t123 = sp.simplify(t10*t25*t34*w1)
    # t131 = sp.simplify(t13*t14*w1*2)
    # t124 = sp.simplify(t122+t123-t131)
    # t139 = sp.simplify(t13*t14*w3)
    # t125 = sp.simplify(-t35+t36+t37+t94-t139)
    # t126 = sp.simplify(X1*t125)
    # t127 = sp.simplify(t49+t50+t51+t52-t64)
    # t128 = sp.simplify(Y1*t127)
    # t129 = sp.simplify(t126+t128-Z1*t124)
    # t130 = sp.simplify(t23*t129*2)
    # t132 = sp.simplify(t13*t26*t45*w1*2)
    # t133 = sp.simplify(t10*t25*t45*w1)
    # t138 = sp.simplify(t13*t14*w2)
    # t134 = sp.simplify(-t46+t47+t48+t113-t138)
    # t135 = sp.simplify(X1*t134)
    # t136 = sp.simplify(-t49-t50+t51+t52+t64)
    # t137 = sp.simplify(Z1*t136)
    # t140 = sp.simplify(X1*s1*t5)
    # t141 = sp.simplify(Y1*s2*t6)
    # t142 = sp.simplify(Z1*s3*t7)
    # t143 = sp.simplify(s1*t1*t5)
    # t144 = sp.simplify(s1*t1*t6)
    # t145 = sp.simplify(s1*t1*t7)
    # print('hello') #5
    # t146 = sp.simplify(s2*t2*t5)
    # t147 = sp.simplify(s2*t2*t6)
    # t148 = sp.simplify(s2*t2*t7)
    # t149 = sp.simplify(s3*t3*t5)
    # t150 = sp.simplify(s3*t3*t6)
    # t151 = sp.simplify(s3*t3*t7)
    # t152 = sp.simplify(X1*s2*w1*w2)
    # t153 = sp.simplify(X1*s3*w1*w3)
    # t154 = sp.simplify(Y1*s1*w1*w2)
    # t155 = sp.simplify(Y1*s3*w2*w3)
    # t156 = sp.simplify(Z1*s1*w1*w3)
    # t157 = sp.simplify(Z1*s2*w2*w3)
    # t158 = sp.simplify(X1*s1*t6*t12)
    # t159 = sp.simplify(X1*s1*t7*t12)
    # t160 = sp.simplify(Y1*s2*t5*t12)
    # t161 = sp.simplify(Y1*s2*t7*t12)
    # t162 = sp.simplify(Z1*s3*t5*t12)
    # t163 = sp.simplify(Z1*s3*t6*t12)
    # t164 = sp.simplify(X1*s2*t9*t10*w3)
    # t165 = sp.simplify(Y1*s3*t9*t10*w1)
    # print('hello') #6
    # t166 = sp.simplify(Z1*s1*t9*t10*w2)
    # t183 = sp.simplify(X1*s3*t9*t10*w2)
    # t184 = sp.simplify(Y1*s1*t9*t10*w3)
    # t185 = sp.simplify(Z1*s2*t9*t10*w1)
    # t186 = sp.simplify(X1*s2*t12*w1*w2)
    # t187 = sp.simplify(X1*s3*t12*w1*w3)
    # t188 = sp.simplify(Y1*s1*t12*w1*w2)
    # t189 = sp.simplify(Y1*s3*t12*w2*w3)
    # t190 = sp.simplify(Z1*s1*t12*w1*w3)
    # t191 = sp.simplify(Z1*s2*t12*w2*w3)
    # t167 = sp.simplify(t140+t141+t142+t143+t144+t145+t146+t147+t148+t149+t150+t151+t152+t153+t154+t155+t156+t157+t158+t159+t160+t161+t162+t163+t164+t165+t166-t183-t184-t185-t186-t187-t188-t189-t190-t191)
    # t168 = sp.simplify(t13*t26*t45*w2*2)
    # t169 = sp.simplify(t10*t25*t45*w2)
    # t170 = sp.simplify(t168+t169)
    # t171 = sp.simplify(t13*t26*t34*w2*2)
    # t172 = sp.simplify(t10*t25*t34*w2)
    # t176 = sp.simplify(t13*t14*w2*2)
    # t173 = sp.simplify(t171+t172-t176)
    # t174 = sp.simplify(-t49+t51+t52+t100-t111)
    # t175 = sp.simplify(X1*t174)
    # t177 = sp.simplify(t13*t24*t26*w2*2)
    # t178 = sp.simplify(t10*t24*t25*w2)
    # t192 = sp.simplify(t13*t14*w1)
    # t179 = sp.simplify(-t97+t98+t99+t112-t192)
    # t180 = sp.simplify(Y1*t179)
    # t181 = sp.simplify(t49+t51+t52-t100+t111)
    # t182 = sp.simplify(Z1*t181)
    # t193 = sp.simplify(t13*t26*t34*w3*2)
    # t194 = sp.simplify(t10*t25*t34*w3)
    # t195 = sp.simplify(t193+t194)
    # t196 = sp.simplify(t13*t26*t45*w3*2)
    # t197 = sp.simplify(t10*t25*t45*w3)
    # t200 = sp.simplify(t13*t14*w3*2)
    # print('hello') #7
    # t198 = sp.simplify(t196+t197-t200)
    # t199 = sp.simplify(t7*t10*t25)
    # t201 = sp.simplify(t13*t24*t26*w3*2)
    # t202 = sp.simplify(t10*t24*t25*w3)
    # t203 = sp.simplify(-t49+t51+t52-t118+t199)
    # t204 = sp.simplify(Y1*t203)
    t205 = sp.simplify(t1*2)
    t206 = sp.simplify(Z1*t29*2)
    t207 = sp.simplify(X1*t32*2)
    t208 = sp.simplify(t205+t206+t207-Y1*t27*2)
    # t209 = sp.simplify(t2*2)
    # t210 = sp.simplify(X1*t53*2)
    # t211 = sp.simplify(Y1*t58*2)
    # t212 = sp.simplify(t209+t210+t211-Z1*t55*2)
    # t213 = sp.simplify(t3*2)
    # t214 = sp.simplify(Y1*t40*2)
    # t215 = sp.simplify(Z1*t43*2)
    # t216 = sp.simplify(t213+t214+t215-X1*t38*2)
    print('jac')
    # jac0 = t14*t65*(X1*r1*w1*2+X1*r2*w2+X1*r3*w3+Y1*r1*w2+Z1*r1*w3+r1*t1*w1*2+r2*t2*w1*2+r3*t3*w1*2+Y1*r3*t5*t12+Y1*r3*t9*t10-Z1*r2*t5*t12-Z1*r2*t9*t10-X1*r2*t12*w2-X1*r3*t12*w3-Y1*r1*t12*w2+Y1*r2*t12*w1*2-Z1*r1*t12*w3+Z1*r3*t12*w1*2+Y1*r3*t5*t10*t11-Z1*r2*t5*t10*t11+X1*r2*t12*w1*w3-X1*r3*t12*w1*w2-Y1*r1*t12*w1*w3+Z1*r1*t12*w1*w2-Y1*r1*t10*t11*w1*w3+Z1*r1*t10*t11*w1*w2-X1*r1*t6*t10*t11*w1-X1*r1*t7*t10*t11*w1+X1*r2*t5*t10*t11*w2+X1*r3*t5*t10*t11*w3+Y1*r1*t5*t10*t11*w2-Y1*r2*t5*t10*t11*w1-Y1*r2*t7*t10*t11*w1+Z1*r1*t5*t10*t11*w3-Z1*r3*t5*t10*t11*w1-Z1*r3*t6*t10*t11*w1+X1*r2*t10*t11*w1*w3-X1*r3*t10*t11*w1*w2+Y1*r3*t10*t11*w1*w2*w3+Z1*r2*t10*t11*w1*w2*w3)-t26*t65*t93*w1*2-t14*t93*t101*(t130+t15*(-X1*t121+Y1*(t46+t47+t48-t13*t14*w2-t12*t14*w1*w3)+Z1*(t35+t36+t37-t13*t14*w3-t10*t25*w1*w2))*2+t18*(t135+t137-Y1*(t132+t133-t13*t14*w1*2))*2)*(1/2)
    # jac0 = sp.simplify(jac0)
    # jac1 = t14*t65*(X1*r2*w1+Y1*r1*w1+Y1*r2*w2*2+Y1*r3*w3+Z1*r2*w3+r1*t1*w2*2+r2*t2*w2*2+r3*t3*w2*2-X1*r3*t6*t12-X1*r3*t9*t10+Z1*r1*t6*t12+Z1*r1*t9*t10+X1*r1*t12*w2*2-X1*r2*t12*w1-Y1*r1*t12*w1-Y1*r3*t12*w3-Z1*r2*t12*w3+Z1*r3*t12*w2*2-X1*r3*t6*t10*t11+Z1*r1*t6*t10*t11+X1*r2*t12*w2*w3-Y1*r1*t12*w2*w3+Y1*r3*t12*w1*w2-Z1*r2*t12*w1*w2-Y1*r1*t10*t11*w2*w3+Y1*r3*t10*t11*w1*w2-Z1*r2*t10*t11*w1*w2-X1*r1*t6*t10*t11*w2+X1*r2*t6*t10*t11*w1-X1*r1*t7*t10*t11*w2+Y1*r1*t6*t10*t11*w1-Y1*r2*t5*t10*t11*w2-Y1*r2*t7*t10*t11*w2+Y1*r3*t6*t10*t11*w3-Z1*r3*t5*t10*t11*w2+Z1*r2*t6*t10*t11*w3-Z1*r3*t6*t10*t11*w2+X1*r2*t10*t11*w2*w3+X1*r3*t10*t11*w1*w2*w3+Z1*r1*t10*t11*w1*w2*w3)-t26*t65*t93*w2*2-t14*t93*t101*(t18*(Z1*(-t35+t94+t95+t96-t13*t14*w3)-Y1*t170+X1*(t97+t98+t99-t13*t14*w1-t10*t25*w2*w3))*2+t15*(t180+t182-X1*(t177+t178-t13*t14*w2*2))*2+t23*(t175+Y1*(t35-t94+t95+t96-t13*t14*w3)-Z1*t173)*2)*(1/2)
    # jac1 = sp.simplify(jac1)
    # jac2 = t14*t65*(X1*r3*w1+Y1*r3*w2+Z1*r1*w1+Z1*r2*w2+Z1*r3*w3*2+r1*t1*w3*2+r2*t2*w3*2+r3*t3*w3*2+X1*r2*t7*t12+X1*r2*t9*t10-Y1*r1*t7*t12-Y1*r1*t9*t10+X1*r1*t12*w3*2-X1*r3*t12*w1+Y1*r2*t12*w3*2-Y1*r3*t12*w2-Z1*r1*t12*w1-Z1*r2*t12*w2+X1*r2*t7*t10*t11-Y1*r1*t7*t10*t11-X1*r3*t12*w2*w3+Y1*r3*t12*w1*w3+Z1*r1*t12*w2*w3-Z1*r2*t12*w1*w3+Y1*r3*t10*t11*w1*w3+Z1*r1*t10*t11*w2*w3-Z1*r2*t10*t11*w1*w3-X1*r1*t6*t10*t11*w3-X1*r1*t7*t10*t11*w3+X1*r3*t7*t10*t11*w1-Y1*r2*t5*t10*t11*w3-Y1*r2*t7*t10*t11*w3+Y1*r3*t7*t10*t11*w2+Z1*r1*t7*t10*t11*w1+Z1*r2*t7*t10*t11*w2-Z1*r3*t5*t10*t11*w3-Z1*r3*t6*t10*t11*w3-X1*r3*t10*t11*w2*w3+X1*r2*t10*t11*w1*w2*w3+Y1*r1*t10*t11*w1*w2*w3)-t26*t65*t93*w3*2-t14*t93*t101*(t18*(Z1*(t46-t113+t114+t115-t13*t14*w2)-Y1*t198+X1*(t49+t51+t52+t118-t7*t10*t25))*2+t23*(X1*(-t97+t112+t116+t117-t13*t14*w1)+Y1*(-t46+t113+t114+t115-t13*t14*w2)-Z1*t195)*2+t15*(t204+Z1*(t97-t112+t116+t117-t13*t14*w1)-X1*(t201+t202-t13*t14*w3*2))*2)*(1/2)
    # jac2 = sp.simplify(jac2)
    # jac3 = sp.simplify(r1*t65-t14*t93*t101*t208*(1/2))
    jac3 = r1*t65-t14*t93*t101*t208*(1/2)
    # jac4 = sp.simplify(r2*t65-t14*t93*t101*t212*(1/2))
    # jac5 = sp.simplify(r3*t65-t14*t93*t101*t216*(1/2))
    # jac6 = t14*t65*(X1*s1*w1*2+X1*s2*w2+X1*s3*w3+Y1*s1*w2+Z1*s1*w3+s1*t1*w1*2+s2*t2*w1*2+s3*t3*w1*2+Y1*s3*t5*t12+Y1*s3*t9*t10-Z1*s2*t5*t12-Z1*s2*t9*t10-X1*s2*t12*w2-X1*s3*t12*w3-Y1*s1*t12*w2+Y1*s2*t12*w1*2-Z1*s1*t12*w3+Z1*s3*t12*w1*2+Y1*s3*t5*t10*t11-Z1*s2*t5*t10*t11+X1*s2*t12*w1*w3-X1*s3*t12*w1*w2-Y1*s1*t12*w1*w3+Z1*s1*t12*w1*w2+X1*s2*t10*t11*w1*w3-X1*s3*t10*t11*w1*w2-Y1*s1*t10*t11*w1*w3+Z1*s1*t10*t11*w1*w2-X1*s1*t6*t10*t11*w1-X1*s1*t7*t10*t11*w1+X1*s2*t5*t10*t11*w2+X1*s3*t5*t10*t11*w3+Y1*s1*t5*t10*t11*w2-Y1*s2*t5*t10*t11*w1-Y1*s2*t7*t10*t11*w1+Z1*s1*t5*t10*t11*w3-Z1*s3*t5*t10*t11*w1-Z1*s3*t6*t10*t11*w1+Y1*s3*t10*t11*w1*w2*w3+Z1*s2*t10*t11*w1*w2*w3)-t14*t101*t167*(t130+t15*(Y1*(t46+t47+t48-t113-t138)+Z1*(t35+t36+t37-t94-t139)-X1*t121)*2+t18*(t135+t137-Y1*(-t131+t132+t133))*2)*(1/2)-t26*t65*t167*w1*2
    # jac6 = sp.simplify(jac6)
    # jac7 = t14*t65*(X1*s2*w1+Y1*s1*w1+Y1*s2*w2*2+Y1*s3*w3+Z1*s2*w3+s1*t1*w2*2+s2*t2*w2*2+s3*t3*w2*2-X1*s3*t6*t12-X1*s3*t9*t10+Z1*s1*t6*t12+Z1*s1*t9*t10+X1*s1*t12*w2*2-X1*s2*t12*w1-Y1*s1*t12*w1-Y1*s3*t12*w3-Z1*s2*t12*w3+Z1*s3*t12*w2*2-X1*s3*t6*t10*t11+Z1*s1*t6*t10*t11+X1*s2*t12*w2*w3-Y1*s1*t12*w2*w3+Y1*s3*t12*w1*w2-Z1*s2*t12*w1*w2+X1*s2*t10*t11*w2*w3-Y1*s1*t10*t11*w2*w3+Y1*s3*t10*t11*w1*w2-Z1*s2*t10*t11*w1*w2-X1*s1*t6*t10*t11*w2+X1*s2*t6*t10*t11*w1-X1*s1*t7*t10*t11*w2+Y1*s1*t6*t10*t11*w1-Y1*s2*t5*t10*t11*w2-Y1*s2*t7*t10*t11*w2+Y1*s3*t6*t10*t11*w3-Z1*s3*t5*t10*t11*w2+Z1*s2*t6*t10*t11*w3-Z1*s3*t6*t10*t11*w2+X1*s3*t10*t11*w1*w2*w3+Z1*s1*t10*t11*w1*w2*w3)-t26*t65*t167*w2*2-t14*t101*t167*(t18*(X1*(t97+t98+t99-t112-t192)+Z1*(-t35+t94+t95+t96-t139)-Y1*t170)*2+t15*(t180+t182-X1*(-t176+t177+t178))*2+t23*(t175+Y1*(t35-t94+t95+t96-t139)-Z1*t173)*2)*(1/2)
    # jac7 = sp.simplify(jac7)
    # jac8 = t14*t65*(X1*s3*w1+Y1*s3*w2+Z1*s1*w1+Z1*s2*w2+Z1*s3*w3*2+s1*t1*w3*2+s2*t2*w3*2+s3*t3*w3*2+X1*s2*t7*t12+X1*s2*t9*t10-Y1*s1*t7*t12-Y1*s1*t9*t10+X1*s1*t12*w3*2-X1*s3*t12*w1+Y1*s2*t12*w3*2-Y1*s3*t12*w2-Z1*s1*t12*w1-Z1*s2*t12*w2+X1*s2*t7*t10*t11-Y1*s1*t7*t10*t11-X1*s3*t12*w2*w3+Y1*s3*t12*w1*w3+Z1*s1*t12*w2*w3-Z1*s2*t12*w1*w3-X1*s3*t10*t11*w2*w3+Y1*s3*t10*t11*w1*w3+Z1*s1*t10*t11*w2*w3-Z1*s2*t10*t11*w1*w3-X1*s1*t6*t10*t11*w3-X1*s1*t7*t10*t11*w3+X1*s3*t7*t10*t11*w1-Y1*s2*t5*t10*t11*w3-Y1*s2*t7*t10*t11*w3+Y1*s3*t7*t10*t11*w2+Z1*s1*t7*t10*t11*w1+Z1*s2*t7*t10*t11*w2-Z1*s3*t5*t10*t11*w3-Z1*s3*t6*t10*t11*w3+X1*s2*t10*t11*w1*w2*w3+Y1*s1*t10*t11*w1*w2*w3)-t26*t65*t167*w3*2-t14*t101*t167*(t18*(Z1*(t46-t113+t114+t115-t138)-Y1*t198+X1*(t49+t51+t52+t118-t199))*2+t23*(X1*(-t97+t112+t116+t117-t192)+Y1*(-t46+t113+t114+t115-t138)-Z1*t195)*2+t15*(t204+Z1*(t97-t112+t116+t117-t192)-X1*(-t200+t201+t202))*2)*(1/2)
    # jac8 = sp.simplify(jac8)
    # jac9 = s1*t65-t14*t101*t167*t208*(1/2)
    # jac10 = s2*t65-t14*t101*t167*t212*(1/2)
    # jac11 = s3*t65-t14*t101*t167*t216*(1/2)
    #
    # jac_np = np.matrix([[jac0,jac1,jac2,jac3,jac4,jac5], \
    #                     [jac6,jac7,jac8,jac9,jac10,jac11]])

    sp.pprint(jac3)
    # sp.pprint(jac4)
    # sp.pprint(jac5)

def myjac():
    sp.init_printing(use_unicode=True)

    pt1,pt2,pt3 = sp.symbols('pt1,pt2,pt3', real=True)

    nr1,nr2,nr3 = sp.symbols('nr1,nr2,nr3', real=True)
    ns1,ns2,ns3 = sp.symbols('ns1,ns2,ns3', real=True)

    r1,r2,r3,t1,t2,t3 = sp.symbols('r1,r2,r3,t1,t2,t3', real=True)
    phi = sp.symbols('phi')
    rod = sp.Matrix([r1,r2,r3])
    phi_ = rod.norm()

    N = sp.Matrix([[0,-r3,r2],[r3,0,-r1],[-r2,r1,0]])
    N = N/phi
    R = sp.eye(3) + sp.simplify(((1-sp.cos(phi))*N*N + sp.sin(phi)*N).subs([(phi_,phi)]))
    # sp.pprint(R)

    R11 = R[0,0] ; R12 = R[0,1] ; R13 = R[0,2]
    R21 = R[1,2] ; R22 = R[1,2] ; R23 = R[1,2]
    R31 = R[2,2] ; R32 = R[2,2] ; R33 = R[2,2]

    pw1 = R11*pt1 + R12*pt2 + R13*pt3 + t1
    pw2 = R21*pt1 + R22*pt2 + R23*pt3 + t2
    pw3 = R31*pt1 + R32*pt2 + R33*pt3 + t3

    lambda_pt = 1/sp.sqrt(pw1**2 + pw2**2 +pw3**2)

    f1 =  nr1*pw1*lambda_pt \
        + nr2*pw2*lambda_pt \
        + nr3*pw3*lambda_pt
    f1 = sp.factor(f1)
    # f2 =  ns1*pw1*lambda_pt \
    #     + ns2*pw2*lambda_pt \
    #     + ns3*pw3*lambda_pt

    # F = sp.Matrix([f1,f2])
    F = sp.Matrix([f1])
    # sp.pprint(F)

    # J = F.jacobian([r1,r2,r3,t1,t2,t3])
    J = sp.simplify(F.jacobian([t1]))
    sp.pprint(J)

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
    # jac_np = np.matrix([[J[0],J[1],J[2],J[3],J[4],J[5]], \
    #                     [J[6],J[7],J[8],J[9],J[10],J[11]]])
    #
    #   [ 3.00203466  0.38069314 -0.96498182 -0.40795141  0.84983699 -0.33369558]
    #   [-1.44983394 -0.09749644 -2.6130425  -0.90655869 -0.33369558  0.25845428]
    return jac_np

### MAIN
if __name__ == '__main__':
    # hisjac()
    myjac()
