#!/usr/bin/python3
# This code is free & open-source, 
# distributed under the terms of GNU General Public 
# License (Version 3, 29 June 2007), 
# https://www.gnu.org/licenses/gpl-3.0.txt .
#   
#   Programmer: Ashkan Shekaari
#
#      Contact: shekaari.theory@gmail.com
#               shekaari@email.kntu.ac.ir
#
# Social Links: https://rt.academia.edu/AshkanShekaari
#               https://orcid.org/0000-0002-7434-467X
#  
# >>> PLEASE, cite the main paper in case of using this code, as follows:
#     A. Shekaari, M. Jafari (2021). "A concise introduction to molecular dynamics simulation: theory and programming". arXiv:2103.16944v1 [cond-mat.mtrl-sci]. doi:10.48550/arXiv.2103.16944.
#
# PURPOSE: Molecular dynamics simulation of a system
# of particles interacting according to Lennard-Jones potential.
#
# Design: Procedural
#
import random, numpy as np
from math import *
#
def PAR():
    """
       Simulation parameters
    """
    __d__ = 3
    NS = 1000
    N = 100
    dt = 1.0E-4
    T_0 = 300.
    sig = 1.
    eps = 1.
    r_ctf = 2.5*sig
    u_at_ctf = 4.*eps*((sig/r_ctf)**12 - (sig/r_ctf)**6)
    du_at_ctf = 24.*eps*(-2.*sig**12/r_ctf**13 \
                   + sig**6/r_ctf**7)
    bs = 10.*sig
    vol = bs**3
    rho = N/vol
    ign = 20
    return __d__, NS, N, dt, T_0, sig, \
    eps, r_ctf, u_at_ctf, du_at_ctf, bs, vol, rho, ign
#------------------------------------------------------
def INIT(__d__, N, bs):
    """
       Initialize positions,
       velocities, 
       and accelerations
    """
    pos = np.zeros([__d__, N])
    vel = np.zeros([__d__, N])
    acc = np.zeros([__d__, N])
    for j in range (N):
        for i in range (__d__):
            pos[i, j] = random.uniform(0, bs)
            vel[i, j] = random.normalvariate(0, 1)
    return pos, vel, acc
#------------------------
def SCL(bs, pos):
    """
       Scale positions, 
       velocities, 
       and accelerations
       to the box size
    """
    pos = pos/bs
    return pos
#-------------------------
def COMREF(__d__, N, pos):
    """
       Translate positions
       to the center-of-mass 
       (com) reference frame
    """
    com = np.zeros ([__d__])
    for i in range(__d__):
        for j in range(N):
            com[i] += pos[i, j]
            com[i] = com[i]/N
    for i in range(__d__):
        pos[i, :] -= com[i]
    return pos
#--------------
def sign(a, b):
    """
       The sign function:
       sign(a, b) returns 'a' 
       with the sign of 'b'
    """
    if b > 0:
       return a
    elif b < 0:
       return -a
    elif b == 0:
       return a
#--------------------------------------------
def FRC(pos, N, __d__, bs, sig, eps, r_ctf, \
               u_at_ctf, du_at_ctf):
    """
       Compute forces, 
       potential energy,
       and virial
    """
    acc = np.zeros([__d__, N])
    pot = np.zeros([N])
    vrl = 0.
    for i in range(N - 1):
        for j in range(i + 1, N):
            R = [pos[0, i] - pos[0, j], pos[1, i] - pos[1, j], \
            pos[2, i] - pos[2, j]]
            if abs(R[0]) > 0.5:
                R[0] -= sign(1., R[0])
            if abs(R[1]) > 0.5:
                R[1] -= sign(1., R[1])
            if abs(R[2]) > 0.5:
                R[2] -= sign(1., R[2])
            r2 = bs*bs*np.dot(R, R)
            if r2 < r_ctf*r_ctf:
                   r1 = sqrt(r2)
                   ri2 = 1./r2
                   ri6 = ri2*ri2*ri2
                   ri12 = ri6*ri6
                   sig6 = sig*sig*sig*sig*sig*sig
                   sig12 = sig6*sig6
                   u = 4.*eps*(sig12*ri12 - sig6*ri6) \
                   - u_at_ctf - r1*du_at_ctf
                   du = 24.*eps*ri2*(2.*sig12*ri12 - sig6*ri6) \
                   + du_at_ctf/r1
                   pot[j] += u
                   vrl -= du*r2
                   for k in range(__d__):
                       acc[k, i] += du*R[k]
                       acc[k, j] -= du*R[k]
    return acc, pot, vrl
#--------------------------------
def TEMP(vel, N, __d__, bs, T_0):
    """
       Compute instantaneous temperature
    """
    kin = np.zeros([N])
    v2 = np.zeros([N])
    for j in range(N):
        for i in range(__d__):
            v2[j] += vel[i, j]*vel[i, j]*bs*bs
        kin[j] = 0.5*v2[j]
    k_AVG = np.mean(kin)
    T_i = 2.*k_AVG/__d__
    B = sqrt(T_0/T_i)
    return T_i, B, k_AVG
#----------------------------------------------
def EVOL(N, __d__, NS, pos, vel, acc, dt, bs, \
                   sig, eps, r_ctf, u_at_ctf, \
                   du_at_ctf, T_0, rho, vol, ign):
    P_S = 0.
    k_S = 0.
    p_S = 0.
    T_S = 0.
    f_tpz = open('tpz.out', 'w')
    f_kpe = open('kpe.out', 'w')
    f_xyz = open('pos.xyz', 'w')
    for step in range(1, NS + 1):
       #----------------------------
       pos[np.where(pos > 0.5)] -= 1
       pos[np.where(pos < -0.5)] += 1
       #-----------------------------------
       acc, pot, vrl = FRC(pos, N, __d__, \
       bs, sig, eps, r_ctf, u_at_ctf, du_at_ctf)
       vrl = -vrl/__d__
       #---------------
       """
          Updating positions
       """
       pos += dt*vel + 0.5*acc*dt*dt
       T_i, B, k_AVG = TEMP(vel, N, __d__, bs, T_0)
       """
          Update velocities
       """
       vel = B*vel + 0.5*dt*acc
       acc, pot, vrl = FRC(pos, N, __d__, \
       bs, sig, eps, r_ctf, u_at_ctf, du_at_ctf)
       vel += 0.5*dt*acc
       T_i, B, k_AVG = TEMP(vel, N, __d__, bs, T_0)
       p_AVG = np.mean(pot)
       etot_AVG = k_AVG + p_AVG
       P = rho*T_i + vrl/vol
       Z = P*vol/(N*T_i)
       f_tpz.write("%d  %e  %e  %e  %e\n" \
       %(step, step*dt, T_i, P, Z))
       f_kpe.write("%d  %e  %e  %e  %e\n" \
       %(step, step*dt, k_AVG, p_AVG, etot_AVG))
       f_xyz.write("%d\n\n" %(step))
       for i in range(N):
           f_xyz.write("%e  %e  %e\n" \
           %(pos[0, i]*bs, pos[1, i]*bs, pos[2, i]*bs))
       if step > ign:
          """
             A condition for ignoring 
             the first "ign" steps 
             in order for reliable 
             statistical averaging.
          """
          P_S += P
          k_S += k_AVG
          p_S += p_AVG
          T_S += T_i
       if step%(NS*0.1) == 0: # To inform that the code is being run
           print(">>>>>  Iteration #  %d  out of %d" %(step, NS))
    return P_S, k_S, p_S, T_S, \
           f_tpz, f_kpe, f_xyz
#------------------------------
def STAVG(NS, ign, P_S, k_S, \
    p_S, T_S, f_tpz, f_kpe, f_xyz):
    """
       Write statistical averages
    """
    f_AVG = open('means.out', 'w')
    f_AVG.write('>>>>>  Statistical Averages  <<<<<\n\n')
    f_AVG.write('     Temperature = %f K\n' %(T_S/(NS - ign)))
    f_AVG.write('  Kinetic energy = %f eps\n' %(k_S/(NS - ign)))
    f_AVG.write('Potential energy = %f eps\n' %(p_S/(NS - ign)))
    f_AVG.write('    Total energy = %f eps\n' %((k_S + p_S)/(NS - ign)))
    f_AVG.write('        Pressure = %f eps/(sig^3)\n' %(P_S/(NS - ign)))
    f_AVG.close()
    f_tpz.close()
    f_kpe.close()
    f_xyz.close()
    print('\n>>>>>  Statistical Averages  <<<<<\n')
    print('     Temperature = %f K' %(T_S/(NS - ign)))
    print('  Kinetic energy = %f eps' %(k_S/(NS - ign)))
    print('Potential energy = %f eps' %(p_S/(NS - ign)))
    print('    Total energy = %f eps' %((k_S + p_S)/(NS - ign)))
    print('        Pressure = %f eps/(sig^3)\n' %(P_S/(NS - ign)))
    return f_AVG
#----------------------------
__d__, NS, N, dt, T_0, sig, \
eps, r_ctf, u_at_ctf, du_at_ctf, bs, \
vol, rho, ign = PAR()
#---------------------------------
pos, vel, acc = INIT(__d__, N, bs)
pos = SCL(bs, pos)
pos = COMREF(__d__, N, pos)
#--------------------------
P_S, k_S, p_S, T_S, \
f_tpz, f_kpe, f_xyz = EVOL(N, \
__d__, NS, pos, vel, acc, dt, bs, sig, eps, \
r_ctf, u_at_ctf, du_at_ctf, T_0, \
rho, vol, ign)
#----------------------
f_AVG = STAVG(NS, ign, \
P_S, k_S, p_S, T_S, \
f_tpz, f_kpe, f_xyz)
