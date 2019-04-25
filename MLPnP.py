### IMPORTS
import numpy as np
import numpy.linalg as npl
import random
from scipy.linalg import null_space
import math
from math import sin, cos, acos, sqrt, pi
import time

### RODRIGUES CONVERSIONS
def rod2rot(rod):
    phi = np.linalg.norm(rod) % (2*pi)
    if phi > np.finfo(float).eps:
        N = np.matrix([ [0.     , -rod[2,0],  rod[1,0] ],
                        [rod[2,0] , 0.     , -rod[0,0] ],
                        [-rod[1,0], rod[0,0] , 0.      ] ])
        N = N / phi
        rot = np.identity(3) + (1-cos(phi))*(N**2) + sin(phi)*N
    else: rot = np.identity(3)
    return rot #np.around(rot,6))

def rot2rod(rot):
    N = [0,0,0]
    phi = acos(max(min((np.trace(rot)-1)/2,1),-1))  # ensure that cos is in [-1,1] after rounds
    if phi > np.finfo(float).eps:
        sc = 1 / (2*sin(phi))
        N[0] = (rot[2,1] - rot[1,2])*sc
        N[1] = (rot[0,2] - rot[2,0])*sc
        N[2] = (rot[1,0] - rot[0,1])*sc
    N = np.matrix(N)/npl.norm(N)*phi
    # Design choice : bearing vector is along +Z (facing camera)
    # This is mainly done to reduce edge effets when phi = pi
    if N[0,2] < 0: N = -1*N
    return N.transpose() #np.around(N,6)


### MLPNP REFINING
# MLPnP Jacobian
# Compute the jacobian associated to a rotation/translation in point pt
def jacobian(pt, nullspace_r, nullspace_s, rot, trans):
    jac = np.zeros((2,6))
    r1 = nullspace_r[0]
    r2 = nullspace_r[1]
    r3 = nullspace_r[2]

    s1 = nullspace_s[0]
    s2 = nullspace_s[1]
    s3 = nullspace_s[2]

    X1 = pt[0]
    Y1 = pt[1]
    Z1 = pt[2]

    w1 = rot[0,0]
    w2 = rot[1,0]
    w3 = rot[2,0]
    t1 = trans[0]
    t2 = trans[1]
    t3 = trans[2]

    t5 = w1*w1
    t6 = w2*w2
    t7 = w3*w3
    t8 = t5+t6+t7
    t9 = sqrt(t8)
    t10 = sin(t9)
    t11 = 1.0/sqrt(t8)
    t12 = cos(t9)
    t13 = t12-1.0
    t14 = 1.0/t8
    t16 = t10*t11*w3
    t17 = t13*t14*w1*w2
    t19 = t10*t11*w2
    t20 = t13*t14*w1*w3
    t24 = t6+t7
    t27 = t16+t17
    t28 = Y1*t27
    t29 = t19-t20
    t30 = Z1*t29
    t31 = t13*t14*t24
    t32 = t31+1.0
    t33 = X1*t32
    t15 = t1-t28+t30+t33
    t21 = t10*t11*w1
    t22 = t13*t14*w2*w3
    t45 = t5+t7
    t53 = t16-t17
    t54 = X1*t53
    t55 = t21+t22
    t56 = Z1*t55
    t57 = t13*t14*t45
    t58 = t57+1.0
    t59 = Y1*t58
    t18 = t2+t54-t56+t59
    t34 = t5+t6
    t38 = t19+t20
    t39 = X1*t38
    t40 = t21-t22
    t41 = Y1*t40
    t42 = t13*t14*t34
    t43 = t42+1.0
    t44 = Z1*t43
    t23 = t3-t39+t41+t44
    t25 = 1.0/(t8**(3.0/2))
    t26 = 1.0/(t8**2)
    t35 = t12*t14*w1*w2
    t36 = t5*t10*t25*w3
    t37 = t5*t13*t26*w3*2.0
    t46 = t10*t25*w1*w3
    t47 = t5*t10*t25*w2
    t48 = t5*t13*t26*w2*2.0
    t49 = t10*t11
    t50 = t5*t12*t14
    t51 = t13*t26*w1*w2*w3*2.0
    t52 = t10*t25*w1*w2*w3
    t60 = t15*t15
    t61 = t18*t18
    t62 = t23*t23
    t63 = t60+t61+t62
    t64 = t5*t10*t25
    t65 = 1.0/sqrt(t63)
    t66 = Y1*r2*t6
    t67 = Z1*r3*t7
    t68 = r1*t1*t5
    t69 = r1*t1*t6
    t70 = r1*t1*t7
    t71 = r2*t2*t5
    t72 = r2*t2*t6
    t73 = r2*t2*t7
    t74 = r3*t3*t5
    t75 = r3*t3*t6
    t76 = r3*t3*t7
    t77 = X1*r1*t5
    t78 = X1*r2*w1*w2
    t79 = X1*r3*w1*w3
    t80 = Y1*r1*w1*w2
    t81 = Y1*r3*w2*w3
    t82 = Z1*r1*w1*w3
    t83 = Z1*r2*w2*w3
    t84 = X1*r1*t6*t12
    t85 = X1*r1*t7*t12
    t86 = Y1*r2*t5*t12
    t87 = Y1*r2*t7*t12
    t88 = Z1*r3*t5*t12
    t89 = Z1*r3*t6*t12
    t90 = X1*r2*t9*t10*w3
    t91 = Y1*r3*t9*t10*w1
    t92 = Z1*r1*t9*t10*w2
    t102 = X1*r3*t9*t10*w2
    t103 = Y1*r1*t9*t10*w3
    t104 = Z1*r2*t9*t10*w1
    t105 = X1*r2*t12*w1*w2
    t106 = X1*r3*t12*w1*w3
    t107 = Y1*r1*t12*w1*w2
    t108 = Y1*r3*t12*w2*w3
    t109 = Z1*r1*t12*w1*w3
    t110 = Z1*r2*t12*w2*w3
    t93 = t66+t67+t68+t69+t70+t71+t72+t73+t74+t75+t76+t77+t78+t79+t80+t81+t82+t83+t84+t85+t86+t87+t88+t89+t90+t91+t92-t102-t103-t104-t105-t106-t107-t108-t109-t110
    t94 = t10*t25*w1*w2
    t95 = t6*t10*t25*w3
    t96 = t6*t13*t26*w3*2.0
    t97 = t12*t14*w2*w3
    t98 = t6*t10*t25*w1
    t99 = t6*t13*t26*w1*2.0
    t100 = t6*t10*t25
    t101 = 1.0/sqrt(t63**3)
    t111 = t6*t12*t14
    t112 = t10*t25*w2*w3
    t113 = t12*t14*w1*w3
    t114 = t7*t10*t25*w2
    t115 = t7*t13*t26*w2*2.0
    t116 = t7*t10*t25*w1
    t117 = t7*t13*t26*w1*2.0
    t118 = t7*t12*t14
    t119 = t13*t24*t26*w1*2.0
    t120 = t10*t24*t25*w1
    t121 = t119+t120
    t122 = t13*t26*t34*w1*2.0
    t123 = t10*t25*t34*w1
    t131 = t13*t14*w1*2.0
    t124 = t122+t123-t131
    t139 = t13*t14*w3
    t125 = -t35+t36+t37+t94-t139
    t126 = X1*t125
    t127 = t49+t50+t51+t52-t64
    t128 = Y1*t127
    t129 = t126+t128-Z1*t124
    t130 = t23*t129*2.0
    t132 = t13*t26*t45*w1*2.0
    t133 = t10*t25*t45*w1
    t138 = t13*t14*w2
    t134 = -t46+t47+t48+t113-t138
    t135 = X1*t134
    t136 = -t49-t50+t51+t52+t64
    t137 = Z1*t136
    t140 = X1*s1*t5
    t141 = Y1*s2*t6
    t142 = Z1*s3*t7
    t143 = s1*t1*t5
    t144 = s1*t1*t6
    t145 = s1*t1*t7
    t146 = s2*t2*t5
    t147 = s2*t2*t6
    t148 = s2*t2*t7
    t149 = s3*t3*t5
    t150 = s3*t3*t6
    t151 = s3*t3*t7
    t152 = X1*s2*w1*w2
    t153 = X1*s3*w1*w3
    t154 = Y1*s1*w1*w2
    t155 = Y1*s3*w2*w3
    t156 = Z1*s1*w1*w3
    t157 = Z1*s2*w2*w3
    t158 = X1*s1*t6*t12
    t159 = X1*s1*t7*t12
    t160 = Y1*s2*t5*t12
    t161 = Y1*s2*t7*t12
    t162 = Z1*s3*t5*t12
    t163 = Z1*s3*t6*t12
    t164 = X1*s2*t9*t10*w3
    t165 = Y1*s3*t9*t10*w1
    t166 = Z1*s1*t9*t10*w2
    t183 = X1*s3*t9*t10*w2
    t184 = Y1*s1*t9*t10*w3
    t185 = Z1*s2*t9*t10*w1
    t186 = X1*s2*t12*w1*w2
    t187 = X1*s3*t12*w1*w3
    t188 = Y1*s1*t12*w1*w2
    t189 = Y1*s3*t12*w2*w3
    t190 = Z1*s1*t12*w1*w3
    t191 = Z1*s2*t12*w2*w3
    t167 = t140+t141+t142+t143+t144+t145+t146+t147+t148+t149+t150+t151+t152+t153+t154+t155+t156+t157+t158+t159+t160+t161+t162+t163+t164+t165+t166-t183-t184-t185-t186-t187-t188-t189-t190-t191
    t168 = t13*t26*t45*w2*2.0
    t169 = t10*t25*t45*w2
    t170 = t168+t169
    t171 = t13*t26*t34*w2*2.0
    t172 = t10*t25*t34*w2
    t176 = t13*t14*w2*2.0
    t173 = t171+t172-t176
    t174 = -t49+t51+t52+t100-t111
    t175 = X1*t174
    t177 = t13*t24*t26*w2*2.0
    t178 = t10*t24*t25*w2
    t192 = t13*t14*w1
    t179 = -t97+t98+t99+t112-t192
    t180 = Y1*t179
    t181 = t49+t51+t52-t100+t111
    t182 = Z1*t181
    t193 = t13*t26*t34*w3*2.0
    t194 = t10*t25*t34*w3
    t195 = t193+t194
    t196 = t13*t26*t45*w3*2.0
    t197 = t10*t25*t45*w3
    t200 = t13*t14*w3*2.0
    t198 = t196+t197-t200
    t199 = t7*t10*t25
    t201 = t13*t24*t26*w3*2.0
    t202 = t10*t24*t25*w3
    t203 = -t49+t51+t52-t118+t199
    t204 = Y1*t203
    t205 = t1*2.0
    t206 = Z1*t29*2.0
    t207 = X1*t32*2.0
    t208 = t205+t206+t207-Y1*t27*2.0
    t209 = t2*2.0
    t210 = X1*t53*2.0
    t211 = Y1*t58*2.0
    t212 = t209+t210+t211-Z1*t55*2.0
    t213 = t3*2.0
    t214 = Y1*t40*2.0
    t215 = Z1*t43*2.0
    t216 = t213+t214+t215-X1*t38*2.0

    jac[0, 0] = t14*t65*(X1*r1*w1*2.0+X1*r2*w2+X1*r3*w3+Y1*r1*w2+Z1*r1*w3+r1*t1*w1*2.0+r2*t2*w1*2.0+r3*t3*w1*2.0+Y1*r3*t5*t12+Y1*r3*t9*t10-Z1*r2*t5*t12-Z1*r2*t9*t10-X1*r2*t12*w2-X1*r3*t12*w3-Y1*r1*t12*w2+Y1*r2*t12*w1*2.0-Z1*r1*t12*w3+Z1*r3*t12*w1*2.0+Y1*r3*t5*t10*t11-Z1*r2*t5*t10*t11+X1*r2*t12*w1*w3-X1*r3*t12*w1*w2-Y1*r1*t12*w1*w3+Z1*r1*t12*w1*w2-Y1*r1*t10*t11*w1*w3+Z1*r1*t10*t11*w1*w2-X1*r1*t6*t10*t11*w1-X1*r1*t7*t10*t11*w1+X1*r2*t5*t10*t11*w2+X1*r3*t5*t10*t11*w3+Y1*r1*t5*t10*t11*w2-Y1*r2*t5*t10*t11*w1-Y1*r2*t7*t10*t11*w1+Z1*r1*t5*t10*t11*w3-Z1*r3*t5*t10*t11*w1-Z1*r3*t6*t10*t11*w1+X1*r2*t10*t11*w1*w3-X1*r3*t10*t11*w1*w2+Y1*r3*t10*t11*w1*w2*w3+Z1*r2*t10*t11*w1*w2*w3)-t26*t65*t93*w1*2.0-t14*t93*t101*(t130+t15*(-X1*t121+Y1*(t46+t47+t48-t13*t14*w2-t12*t14*w1*w3)+Z1*(t35+t36+t37-t13*t14*w3-t10*t25*w1*w2))*2.0+t18*(t135+t137-Y1*(t132+t133-t13*t14*w1*2.0))*2.0)*(1.0/2.0)
    jac[0, 1] = t14*t65*(X1*r2*w1+Y1*r1*w1+Y1*r2*w2*2.0+Y1*r3*w3+Z1*r2*w3+r1*t1*w2*2.0+r2*t2*w2*2.0+r3*t3*w2*2.0-X1*r3*t6*t12-X1*r3*t9*t10+Z1*r1*t6*t12+Z1*r1*t9*t10+X1*r1*t12*w2*2.0-X1*r2*t12*w1-Y1*r1*t12*w1-Y1*r3*t12*w3-Z1*r2*t12*w3+Z1*r3*t12*w2*2.0-X1*r3*t6*t10*t11+Z1*r1*t6*t10*t11+X1*r2*t12*w2*w3-Y1*r1*t12*w2*w3+Y1*r3*t12*w1*w2-Z1*r2*t12*w1*w2-Y1*r1*t10*t11*w2*w3+Y1*r3*t10*t11*w1*w2-Z1*r2*t10*t11*w1*w2-X1*r1*t6*t10*t11*w2+X1*r2*t6*t10*t11*w1-X1*r1*t7*t10*t11*w2+Y1*r1*t6*t10*t11*w1-Y1*r2*t5*t10*t11*w2-Y1*r2*t7*t10*t11*w2+Y1*r3*t6*t10*t11*w3-Z1*r3*t5*t10*t11*w2+Z1*r2*t6*t10*t11*w3-Z1*r3*t6*t10*t11*w2+X1*r2*t10*t11*w2*w3+X1*r3*t10*t11*w1*w2*w3+Z1*r1*t10*t11*w1*w2*w3)-t26*t65*t93*w2*2.0-t14*t93*t101*(t18*(Z1*(-t35+t94+t95+t96-t13*t14*w3)-Y1*t170+X1*(t97+t98+t99-t13*t14*w1-t10*t25*w2*w3))*2.0+t15*(t180+t182-X1*(t177+t178-t13*t14*w2*2.0))*2.0+t23*(t175+Y1*(t35-t94+t95+t96-t13*t14*w3)-Z1*t173)*2.0)*(1.0/2.0)
    jac[0, 2] = t14*t65*(X1*r3*w1+Y1*r3*w2+Z1*r1*w1+Z1*r2*w2+Z1*r3*w3*2.0+r1*t1*w3*2.0+r2*t2*w3*2.0+r3*t3*w3*2.0+X1*r2*t7*t12+X1*r2*t9*t10-Y1*r1*t7*t12-Y1*r1*t9*t10+X1*r1*t12*w3*2.0-X1*r3*t12*w1+Y1*r2*t12*w3*2.0-Y1*r3*t12*w2-Z1*r1*t12*w1-Z1*r2*t12*w2+X1*r2*t7*t10*t11-Y1*r1*t7*t10*t11-X1*r3*t12*w2*w3+Y1*r3*t12*w1*w3+Z1*r1*t12*w2*w3-Z1*r2*t12*w1*w3+Y1*r3*t10*t11*w1*w3+Z1*r1*t10*t11*w2*w3-Z1*r2*t10*t11*w1*w3-X1*r1*t6*t10*t11*w3-X1*r1*t7*t10*t11*w3+X1*r3*t7*t10*t11*w1-Y1*r2*t5*t10*t11*w3-Y1*r2*t7*t10*t11*w3+Y1*r3*t7*t10*t11*w2+Z1*r1*t7*t10*t11*w1+Z1*r2*t7*t10*t11*w2-Z1*r3*t5*t10*t11*w3-Z1*r3*t6*t10*t11*w3-X1*r3*t10*t11*w2*w3+X1*r2*t10*t11*w1*w2*w3+Y1*r1*t10*t11*w1*w2*w3)-t26*t65*t93*w3*2.0-t14*t93*t101*(t18*(Z1*(t46-t113+t114+t115-t13*t14*w2)-Y1*t198+X1*(t49+t51+t52+t118-t7*t10*t25))*2.0+t23*(X1*(-t97+t112+t116+t117-t13*t14*w1)+Y1*(-t46+t113+t114+t115-t13*t14*w2)-Z1*t195)*2.0+t15*(t204+Z1*(t97-t112+t116+t117-t13*t14*w1)-X1*(t201+t202-t13*t14*w3*2.0))*2.0)*(1.0/2.0)
    jac[0, 3] = r1*t65-t14*t93*t101*t208*(1.0/2.0)
    jac[0, 4] = r2*t65-t14*t93*t101*t212*(1.0/2.0)
    jac[0, 5] = r3*t65-t14*t93*t101*t216*(1.0/2.0)
    jac[1, 0] = t14*t65*(X1*s1*w1*2.0+X1*s2*w2+X1*s3*w3+Y1*s1*w2+Z1*s1*w3+s1*t1*w1*2.0+s2*t2*w1*2.0+s3*t3*w1*2.0+Y1*s3*t5*t12+Y1*s3*t9*t10-Z1*s2*t5*t12-Z1*s2*t9*t10-X1*s2*t12*w2-X1*s3*t12*w3-Y1*s1*t12*w2+Y1*s2*t12*w1*2.0-Z1*s1*t12*w3+Z1*s3*t12*w1*2.0+Y1*s3*t5*t10*t11-Z1*s2*t5*t10*t11+X1*s2*t12*w1*w3-X1*s3*t12*w1*w2-Y1*s1*t12*w1*w3+Z1*s1*t12*w1*w2+X1*s2*t10*t11*w1*w3-X1*s3*t10*t11*w1*w2-Y1*s1*t10*t11*w1*w3+Z1*s1*t10*t11*w1*w2-X1*s1*t6*t10*t11*w1-X1*s1*t7*t10*t11*w1+X1*s2*t5*t10*t11*w2+X1*s3*t5*t10*t11*w3+Y1*s1*t5*t10*t11*w2-Y1*s2*t5*t10*t11*w1-Y1*s2*t7*t10*t11*w1+Z1*s1*t5*t10*t11*w3-Z1*s3*t5*t10*t11*w1-Z1*s3*t6*t10*t11*w1+Y1*s3*t10*t11*w1*w2*w3+Z1*s2*t10*t11*w1*w2*w3)-t14*t101*t167*(t130+t15*(Y1*(t46+t47+t48-t113-t138)+Z1*(t35+t36+t37-t94-t139)-X1*t121)*2.0+t18*(t135+t137-Y1*(-t131+t132+t133))*2.0)*(1.0/2.0)-t26*t65*t167*w1*2.0
    jac[1, 1] = t14*t65*(X1*s2*w1+Y1*s1*w1+Y1*s2*w2*2.0+Y1*s3*w3+Z1*s2*w3+s1*t1*w2*2.0+s2*t2*w2*2.0+s3*t3*w2*2.0-X1*s3*t6*t12-X1*s3*t9*t10+Z1*s1*t6*t12+Z1*s1*t9*t10+X1*s1*t12*w2*2.0-X1*s2*t12*w1-Y1*s1*t12*w1-Y1*s3*t12*w3-Z1*s2*t12*w3+Z1*s3*t12*w2*2.0-X1*s3*t6*t10*t11+Z1*s1*t6*t10*t11+X1*s2*t12*w2*w3-Y1*s1*t12*w2*w3+Y1*s3*t12*w1*w2-Z1*s2*t12*w1*w2+X1*s2*t10*t11*w2*w3-Y1*s1*t10*t11*w2*w3+Y1*s3*t10*t11*w1*w2-Z1*s2*t10*t11*w1*w2-X1*s1*t6*t10*t11*w2+X1*s2*t6*t10*t11*w1-X1*s1*t7*t10*t11*w2+Y1*s1*t6*t10*t11*w1-Y1*s2*t5*t10*t11*w2-Y1*s2*t7*t10*t11*w2+Y1*s3*t6*t10*t11*w3-Z1*s3*t5*t10*t11*w2+Z1*s2*t6*t10*t11*w3-Z1*s3*t6*t10*t11*w2+X1*s3*t10*t11*w1*w2*w3+Z1*s1*t10*t11*w1*w2*w3)-t26*t65*t167*w2*2.0-t14*t101*t167*(t18*(X1*(t97+t98+t99-t112-t192)+Z1*(-t35+t94+t95+t96-t139)-Y1*t170)*2.0+t15*(t180+t182-X1*(-t176+t177+t178))*2.0+t23*(t175+Y1*(t35-t94+t95+t96-t139)-Z1*t173)*2.0)*(1.0/2.0)
    jac[1, 2] = t14*t65*(X1*s3*w1+Y1*s3*w2+Z1*s1*w1+Z1*s2*w2+Z1*s3*w3*2.0+s1*t1*w3*2.0+s2*t2*w3*2.0+s3*t3*w3*2.0+X1*s2*t7*t12+X1*s2*t9*t10-Y1*s1*t7*t12-Y1*s1*t9*t10+X1*s1*t12*w3*2.0-X1*s3*t12*w1+Y1*s2*t12*w3*2.0-Y1*s3*t12*w2-Z1*s1*t12*w1-Z1*s2*t12*w2+X1*s2*t7*t10*t11-Y1*s1*t7*t10*t11-X1*s3*t12*w2*w3+Y1*s3*t12*w1*w3+Z1*s1*t12*w2*w3-Z1*s2*t12*w1*w3-X1*s3*t10*t11*w2*w3+Y1*s3*t10*t11*w1*w3+Z1*s1*t10*t11*w2*w3-Z1*s2*t10*t11*w1*w3-X1*s1*t6*t10*t11*w3-X1*s1*t7*t10*t11*w3+X1*s3*t7*t10*t11*w1-Y1*s2*t5*t10*t11*w3-Y1*s2*t7*t10*t11*w3+Y1*s3*t7*t10*t11*w2+Z1*s1*t7*t10*t11*w1+Z1*s2*t7*t10*t11*w2-Z1*s3*t5*t10*t11*w3-Z1*s3*t6*t10*t11*w3+X1*s2*t10*t11*w1*w2*w3+Y1*s1*t10*t11*w1*w2*w3)-t26*t65*t167*w3*2.0-t14*t101*t167*(t18*(Z1*(t46-t113+t114+t115-t138)-Y1*t198+X1*(t49+t51+t52+t118-t199))*2.0+t23*(X1*(-t97+t112+t116+t117-t192)+Y1*(-t46+t113+t114+t115-t138)-Z1*t195)*2.0+t15*(t204+Z1*(t97-t112+t116+t117-t192)-X1*(-t200+t201+t202))*2.0)*(1.0/2.0)
    jac[1, 3] = s1*t65-t14*t101*t167*t208*(1.0/2.0)
    jac[1, 4] = s2*t65-t14*t101*t167*t212*(1.0/2.0)
    jac[1, 5] = s3*t65-t14*t101*t167*t216*(1.0/2.0)

    return jac

# Residuals and jacobians for all points
# Compute the residuals and jacobians for a set of points, a transformation x and the corresponding nullspaces
def residuals_and_jacs(pts, nullspace_r, nullspace_s, x):
    nb_obs = pts.shape[1]
    nb_unknowns = 6
    w = x[0:3]
    R = rod2rot(w)
    T = np.matrix(x[3:6])

    ii = 0
    r = np.zeros((2*nb_obs,1));
    jacobians = np.zeros((2*nb_obs,nb_unknowns))

    for i in range(nb_obs):
        # pt = R*pts[i] + T
        pt = R @ pts[:,i] + T
        pt /= np.linalg.norm(pt)
        # r = nullspace[i]^T * pt
        r[ii  ,0] = nullspace_r[:,i] @ pt
        r[ii+1,0] = nullspace_s[:,i] @ pt
        # jacs
        jac = jacobian(pts[:,i],nullspace_r[:,i], nullspace_s[:,i], w, T)
        jacobians[ii  ,:] = jac[0,:]
        jacobians[ii+1,:] = jac[1,:]
        ii += 2
    return jacobians, r

# Gauss Newton optimization for MLPnP solution
# Refine the 6D transformation x, from a set a points, the corresponding nullspaces, the covariance matrix Kll, and the initial guess for x
def refine_gauss_newton(x, pts, nullspace_r, nullspace_s, Kll, use_cov):
    nb_obs = pts.shape[1]
    nb_unknowns = 6
    assert ((2 * nb_obs - 6) > 0)

    # Set matrices
    r = np.zeros(2*nb_obs)
    rd = np.zeros(2*nb_obs)
    dx = np.zeros((nb_unknowns, 1))
    eyeMat = np.identity(nb_unknowns)

    iter = 0
    stop = False
    max_it = 5
    eps = 1e-6

    while iter < max_it and not stop:
        jacs, r = residuals_and_jacs(pts, nullspace_r, nullspace_s, x)
        if use_cov: JacTSKll = jacs.transpose().dot(Kll)
        else: JacTSKll = jacs.transpose()
        # Design matrix
        N = JacTSKll.dot(jacs)
        # Get system
        g = JacTSKll.dot(r)

        # Solve
        # chol = npl.cholesky(N)
        # dx = npl.solve(chol,g)
        dx = npl.pinv(N).dot(g)
        if np.amax(np.absolute(np.asarray(dx))) > 5. or np.amin(np.absolute(np.asarray(dx))) > 1.: break
        dl = jacs.dot(dx)

        # Update transformation vector
        if np.amax(np.absolute(np.asarray(dl))) < eps:
            stop = True
            x = x - dx
            break
        else:
            x = x - dx
        iter += 1

    # Statistics
    Qxx = npl.inv(N)
    Qldld = jacs.dot(Qxx).dot(jacs.transpose())
    return x


### MLPNP
# Estimate 4x4 transform matrix (object to camera) from a set of N 3D points (in the object coordinate system),
# and the corresponding bearing vectors (image rays) and its covariance matrix (size 9*N) if available
def mlpnp(pts, v, cov = None):
    assert pts.shape[1] > 5
    use_cov = (cov is not None)
    # Definitions
    nb_pts = pts.shape[1]
    nullspace_r = np.zeros((3,nb_pts))
    nullspace_s = np.zeros((3,nb_pts))
    cov_reduced = np.zeros((2,2,nb_pts))

    ### TODO : planar case

    # Compute nullspaces
    for i in range(nb_pts):
        null_2d = null_space(v[:,i].transpose())
        nullspace_r[:,i] = null_2d[:,0]
        nullspace_s[:,i] = null_2d[:,1]
        if use_cov:
            cov_reduced[:,:,i] = npl.inv(null_2d.transpose().dot(np.reshape(cov[:,i],(3,3)).dot(null_2d)))

    # Stochastic model
    P = np.identity(2*nb_pts)
    # Design matrix
    A = np.zeros((2*nb_pts,12))
    for i in range(nb_pts):
        # Covariance
        if use_cov: P[2*i:2*(i+1),2*i:2*(i+1)] = cov_reduced[:,:,i]
        # r11
        A[2*i  , 0] = nullspace_r[0,i] * pts[0,i]
        A[2*i+1, 0] = nullspace_s[0,i] * pts[0,i]
        # r12
        A[2*i  , 1] = nullspace_r[0,i] * pts[1,i]
        A[2*i+1, 1] = nullspace_s[0,i] * pts[1,i]
        # r13
        A[2*i  , 2] = nullspace_r[0,i] * pts[2,i]
        A[2*i+1, 2] = nullspace_s[0,i] * pts[2,i]
        # r21
        A[2*i  , 3] = nullspace_r[1,i] * pts[0,i]
        A[2*i+1, 3] = nullspace_s[1,i] * pts[0,i]
        # r22
        A[2*i  , 4] = nullspace_r[1,i] * pts[1,i]
        A[2*i+1, 4] = nullspace_s[1,i] * pts[1,i]
        # r23
        A[2*i  , 5] = nullspace_r[1,i] * pts[2,i]
        A[2*i+1, 5] = nullspace_s[1,i] * pts[2,i]
        # r31
        A[2*i  , 6] = nullspace_r[2,i] * pts[0,i]
        A[2*i+1, 6] = nullspace_s[2,i] * pts[0,i]
        # r32
        A[2*i  , 7] = nullspace_r[2,i] * pts[1,i]
        A[2*i+1, 7] = nullspace_s[2,i] * pts[1,i]
        # r33
        A[2*i  , 8] = nullspace_r[2,i] * pts[2,i]
        A[2*i+1, 8] = nullspace_s[2,i] * pts[2,i]
        # t1
        A[2*i  , 9] = nullspace_r[0,i]
        A[2*i+1, 9] = nullspace_s[0,i]
        # t2
        A[2*i  ,10] = nullspace_r[1,i]
        A[2*i+1,10] = nullspace_s[1,i]
        # t3
        A[2*i  ,11] = nullspace_r[2,i]
        A[2*i+1,11] = nullspace_s[2,i]

    # N = AtPAx
    N = A.transpose().dot(P).dot(A)
    # SVD of N
    _,_,V = npl.svd(A)
    V = V.transpose()
    R_tmp = np.reshape(V[0:9,-1],(3,3))
    # SVD to find the best rotation matrix in the Frobenius sense
    Ur,_,VHr = npl.svd(R_tmp.transpose())
    R = Ur.dot(VHr)
    if npl.det(R) < 0: R = -1*R
    # Recover translation
    t = V[9:12,-1]
    t /= ( npl.norm(R_tmp[:,0],axis = 0)*npl.norm(R_tmp[:,1])*npl.norm(R_tmp[:,2]) )**(1./3)
    t = R.dot(t)

    # Create transformation matrices to determine translation sign
    transform_1 = np.empty((4,4))
    transform_1[0:3,0:3] = R
    transform_1[0:3,3] = t
    transform_1[3,:] = [0,0,0,1]
    transform_2 = np.copy(transform_1)
    transform_2[0:3,3] = -t
    transform_1 = np.matrix(transform_1)
    transform_2 = np.matrix(transform_2)
    transform_1_inv = npl.inv(transform_1)
    transform_2_inv = npl.inv(transform_2)

    # find the best solution with 6 correspondences
    diff1 = 0
    diff2 = 0
    for i in range(6):
        pt4 = np.concatenate((pts[:,i],np.matrix(1)), axis=0)
        testres1 = transform_1_inv @ pt4
        testres2 = transform_2_inv @ pt4
        testres1 = testres1[0:3] / npl.norm(testres1[0:3])
        testres2 = testres2[0:3] / npl.norm(testres2[0:3])
        diff1 += 1-np.dot(testres1.transpose(), v[:,i])
        diff2 += 1-np.dot(testres2.transpose(), v[:,i])

    if diff1 <= diff2: transform = transform_1[0:3,0:4]
    else: transform = transform_2[0:3,0:4]

    x = np.matrix(np.concatenate((rot2rod(transform[:,0:3]),transform[:,3])))

    # Refine with Gauss Newton
    x_gn = [0]
    tic = time.time()
    x_gn = refine_gauss_newton(x, pts, nullspace_r, nullspace_s, P, use_cov)
    tac = time.time()
    print('gauss newton :',tac-tic)
    return np.around(x,10), np.around(x_gn,10)


### MAIN
if __name__ == '__main__':
    # Convert pixel coordinate points into rays (unitary bearing vectors)
    # K : camera matrix (3x3)
    # pix : each collumn of the matrix is a pixel to transform into ray
    def pix2rays(K, pix):
        # add z coordinate if not already existing (=1, on image plane)
        if pix.shape[0] == 2:
            pix = np.concatenate((pix,np.ones((pix.shape[1],1)).transpose()), axis = 0)
        rays = npl.inv(K[0:3,0:3])*pix
        rays[0,:] = -rays[0,:] # invert x coordinate because U (in image plane) is along -X (in camera frame)
        rays /= npl.norm(rays,axis = 0)
        return rays

    print("^_^ My name is MLPnP.py ^_^ \n")

    # Intrinsics matrix
    K = np.matrix('640 1 320 ; 0 480 240 ; 0 0 1')

    debug = True
    if debug:
        nb_iter = 1
        display = True
        randomize = False
    else:
        nb_iter = 500
        display = False
        randomize = True

    count_ok = 0
    count_ko = 0
    for i in range(nb_iter):
        # Ground truth transformation from cam to world
        if randomize:
            phi = random.uniform(0, 2*pi)
            axis = np.matrix(np.random.random((3,1)))
            trans = np.matrix(np.random.random((3,1)))
        else:
            phi = pi  % (2*pi)
            axis = np.matrix('0 0 1').transpose()
            trans = np.matrix('0 0 -1').transpose()

        rod = phi*axis/npl.norm(axis)
        # print(np.around(rod2rot(rod),2)) ### DEBUG
        x_gt = np.concatenate((rod,trans), axis = 0)

        nb_pts = 10 # Number of points to generate
        # Sample random points in image space
        pix = np.concatenate((np.random.randint(0,640,(1,nb_pts)), np.random.randint(0,480,(1,nb_pts))), axis = 0)
        # Convert those pixels to rays adding gaussian noise
        rays = pix2rays(K,pix)

        # Sample random distances for world coordinates
        min_dist = 2
        max_dist = 10
        norms = np.random.uniform(min_dist,max_dist,(nb_pts))

        # Compute 3D points positions in camera coordinates
        noise_sd = 0.001
        cam_pts = rays * np.diag(norms) + np.random.normal(0,noise_sd,rays.shape)

        # Convert to world coordinates
        world_pts = rod2rot(x_gt[0:3]) @ cam_pts + np.repeat(x_gt[3:6],nb_pts, axis = 1)

        # Apply PnP
        tic = time.time()
        x, x_gn = mlpnp(world_pts, rays)
        tac = time.time()
        print('overall :', tac-tic)
        if display:
            print('x_gt  :\n', x_gt)
            print('x_pnp :\n', x)
            print('x_gn  :\n', x_gn)
        if npl.norm(x_gt-x) > npl.norm(x_gt-x_gn):
            count_ko += 1
            # print('x_gt  :\n', x_gt)
            # print('x_pnp :\n', x)
        else:
            count_ok += 1
    print('ok :', count_ok, '\nko :', count_ko)

# pts = np.matrix(np.random.rand(3,nb_pts))
# cov = np.random.rand(9,nb_pts)
