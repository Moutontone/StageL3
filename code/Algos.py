#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module documentation goes here
   and here
   and ...

Created on Fri Jul 8 2022
@author: Armand
"""
from random import random, seed, shuffle
import numpy as np
import matplotlib.pyplot as plt
import time

from Modeles import Modele

"""
    Les différents algorithms
    - Random
    - Offline
    - Kl_ucb
    - Kl_ucb_mod
    - ?
"""

class Algo():
    """ class Algo
        Description...
        paramètre :
            - Ut : vecteur des U à chaque t (U: estimations optimistes de  l'attirance à t)
            - Nt : vecteur des N à chaque t (N: nombres d'utilisations des items à t)
            - Wn : etc.
            - Gt
            - At
            - Ct
            - wt
            - duree
    """
    def __init__(self):
        self.At = []
        self.Ct = []
        self.wt = []
        self.Gt = []
        self.duree = -1
        # self.Ut = []
        # self.Nt = []
        # self.Wn = []

    def duree_start(self):
        self.duree = time.time()

    def duree_stop(self):
        self.duree = time.time() - self.duree

    def run(self, b, T, aff = False):
            print("Algo.run() ne fait rien du tout.")
            return

class Random(Algo):
    """docstring for Random."""

    def __init__(self):
        super(Random, self).__init__()

    def init_param(self):
        self.At = []
        self.Ct = []
        self.wt = []
        self.Gt = []

    def add_param(self, g, A, C, w):
        self.Gt.append(g)
        self.At.append([a for a in A])
        self.Ct.append([c for c in C])
        self.wt.append(w)

    def run(self, b, T, aff = False):
        self.init_param()
        self.duree_start()
        se = sousEnsemblesk(b.k, b.n)
        for t in range(T):
            if aff: print(f"Temps {t} sur {T}")
            A = se[int(random() * len(se))]
            C, w = b.recommander(A)
            g = b.G[w] if w > -1 else 0
            self.add_param(g, A, C, w)
        self.duree_stop()
        return

class Offline(Algo):
    """docstring for Offline."""

    def __init__(self):
        super(Offline, self).__init__()

    def init_param(self):
        self.At = []
        self.Ct = []
        self.wt = []
        self.Gt = []

    def add_param(self, g, A, C, w):
        self.Gt.append(g)
        self.At.append([a for a in A])
        self.Ct.append([c for c in C])
        self.wt.append(w)

    def run(self, b, T, aff = False):
        self.init_param()
        self.duree_start()
        if aff : print("recherche de la meilleure solution")
        A = argmaxGtot(b)
        if aff : print(f"solution trouvé en {time.time() - self.duree}")
        for t in range(T):
            C, w = b.recommander(A)
            g = b.G[w] if w > -1 else 0
            self.add_param(g, A, C, w)
        self.duree_stop()
        return

class Kl_ucb(Algo):
    """ Algo Kl_ucb
        Description...
        paramètre de run(b, T) :
            - b : le bandit, instance du Modèle de Modele.py
            - T : horizon du temps
            - (optionel) aff = False : condition d'affichage des étapes
    """

    def __init__(self):
        super(Kl_ucb, self).__init__()

    def init_param(self, b, U, N, W):
        self.Ut = [[u for u in U]]
        self.Nt = [[n for n in N]]
        self.Wn = [[w for w in W]]
        self.Gt = [-1]
        self.At = [[-1]]
        self.Ct = [[-1 for _ in range(b.k)]]
        self.wt = [-1]

    def add_param(self, U, N, W, g, A, C, w):
        self.Ut.append([u for u in U])
        self.Nt.append([n for n in N])
        self.Wn.append([w for w in W])
        self.Gt.append(g)
        self.At.append([a for a in A])
        self.Ct.append([c for c in C])
        self.wt.append(w)

    def run(self, b, T, aff = False):
        # TODO: Faire une meilleure initialisiation en fonction de ce qui est décrit dans l'article
        # initialisiation
        self.duree_start()
        U = [1 for _ in range(b.n)] # liste des moyennes optimistes
        N = [1 for _ in range(b.n)] # nombre d'informations sur les éléments à t
        Np = [0 for _ in range(b.n)] # nombre d'informations sur les éléments à t-1
        W = [round(random()) for _ in range(b.n)] # estimation de W
        G = [] # gains au cours du temps
        self.init_param(b, U, N, W)
        # ...
        for t in range(1, T):
            Np = [n for n in N]
            if aff: print("Temps ", t)
            # maj des estimateur optimistes
            U = UCBKL(W, Np, t)
            if aff: print(f"{t} > Np : {Np}")
            if aff: print(f"{t} > U : {U}")
            if aff: print(f"{t} > W : {W}")
            # calcul des elements à présenter
            A = argmax_klucb(b, U)
            # recommandation et mise à jour des estimateurs
            C, w = b.recommander(A)
            #G.append(b.G[w] if w > -1 else 0)
            g = b.G[w] if w > -1 else 0
            if aff: print(f"{t} > A {A}")
            if aff: print(f"{t} > C = {C}, w = {w}")
            for k in range(c_last(C, b.k) + 1):
                e = A[k]
                N[e] += 1
                if aff: print(f"e {e}, W[e] {W[e]}, Np[e] {Np[e]}, N[e] {N[e]}, C[k] {C[k]} -> ", end = "")
                W[e] = (Np[e] * W[e]  + C[k])/N[e]
                if aff: print(W[e])
            self.add_param(U, N, W, g, A , C, w)
        self.duree_stop()
        return

class Kl_ucb_mod(Algo):
    """ docstring """

    def __init__(self):
        super(Kl_ucb, self).__init__()

    def init_param(self, b, U, N, W):
        self.Ut = [[u for u in U]]
        self.Nt = [[n for n in N]]
        self.Wn = [[w for w in W]]
        self.Gt = [-1]
        self.At = [[-1]]
        self.Ct = [[-1 for _ in range(b.k)]]
        self.wt = [-1]

    def add_param(self, U, N, W, g, A, C, w):
        self.Ut.append([u for u in U])
        self.Nt.append([n for n in N])
        self.Wn.append([w for w in W])
        self.Gt.append(g)
        self.At.append([a for a in A])
        self.Ct.append([c for c in C])
        self.wt.append(w)

    def run(self, b, T, aff = False):
        # TODO: Faire une meilleure initialisiation en fonction de ce qui est décrit dans l'article
        # initialisiation
        self.duree_start()
        U = [1 for _ in range(b.n)] # liste des moyennes optimistes
        N = [1 for _ in range(b.n)] # nombre d'informations sur les éléments à t
        Np = [0 for _ in range(b.n)] # nombre d'informations sur les éléments à t-1
        W = [round(random()) for _ in range(b.n)] # estimation de W
        G = [] # gains au cours du temps
        self.init_param(b, U, N, W)
        # ...
        for t in range(1, T):
            Np = [n for n in N]
            if aff: print("Temps ", t)
            # maj des estimateur optimistes
            U = UCBKL(W, Np, t)
            if aff: print(f"{t} > Np : {Np}")
            if aff: print(f"{t} > U : {U}")
            if aff: print(f"{t} > W : {W}")
            # calcul des elements à présenter
            A = argmax_klucb(b, U)
            # recommandation et mise à jour des estimateurs
            C, w = b.recommander(A)
            #G.append(b.G[w] if w > -1 else 0)
            g = b.G[w] if w > -1 else 0
            if aff: print(f"{t} > A {A}")
            if aff: print(f"{t} > C = {C}, w = {w}")
            for k in range(c_last(C, b.k) + 1):
                e = A[k]
                N[e] += 1
                if aff: print(f"e {e}, W[e] {W[e]}, Np[e] {Np[e]}, N[e] {N[e]}, C[k] {C[k]} -> ", end = "")
                W[e] = (Np[e] * W[e]  + C[k])/N[e]
                if aff: print(W[e])
            self.add_param(U, N, W, g, A , C, w)
        self.duree_stop()
        return


"""
    Les fonctions à utiliser dans les algorithms
"""
# dernier item avec un clique ou k si 0 cliques
def c_last(C, k):
    i = len(C) -1
    while C[i] != 1 and i >= 0 :
        i -= 1
    if i < 0:
        return k-1
    return i

# trouver la liste d'item a proposer
def argmax_klucb(b, U):
    V = [1 - i/b.k for i in range(b.k)]
    f = lambda x: f_katariya(x, U, V)
    maxS = []
    maxG = 0
    for S in sousEnsemblesk(b.k, b.n):
        # print(f"S : {S}", end = "")
        g = f(S)
        # print(f" -> {g}")
        if  g > maxG:
            maxG = g
            maxS = S
    return maxS

def f_katariya(A, W, V):
    p = 1
    for k in range(len(A)):
        p *= (1-V[k]*W[A[k]])
    return 1 - p


# pour creer les sous ensembles de taille k de [n]
def sousEnsemblesk(k, n):
    E = []
    rec_ssE(E,k,n-1,[])
    return E[::-1]

def rec_ssE(E,k,n,R):
    if n < 0:
        return
    A = [e for e in R]
    A.append(n)
    if len(A) == k:
        E.append(A)
    else:
        rec_ssE(E, k, n-1, A)
    rec_ssE(E, k, n-1, R)

# calcul de U en utilisant D_kl
def UCBKL(W, Np, t):
    U = [zero(W[i], Np[i], t) for i in range(len(W))]
    return U

def DKL(p,q):
    return p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def zero(w,n,t, e = 0.001):
    if w == 0:
        w += e/10
    # on cherche un zero de la fonction :
    kl = lambda q: n*DKL(w,q) - np.log(t) - 3* np.log(np.log(t))
    if t == 1:
        kl = lambda q: n*DKL(w,q) - np.log(t) - 3* np.log(t)
    # dkl = lambda q: n*(-w/q + (1-w)/(1-q))
    # par dichotomie
    # print(f"w {w}\nn {n}\nt {t}")
    a, b = w, 1
    c = (a+b)/2
    # while kl(c) > 0 or kl(c) < epsilon:
    while b-a > e:
        # print(f"[a, b] [{a}, {b}]")
        # print(f"c {c} -> kl(c) {kl(c)}")
        if kl(c) < 0:
            a = c
        else:
            b = c
        c = (a+b)/2
    return a

# Calcul du gain a maximiser d'une proposition A
def argmaxGtot(b):
    maxS = []
    maxG = 0
    for S in sousEnsemblesk(b.k, b.n):
        # print(f"S : {S}", end = "")
        g = Gtot(b, S)
        # print(f" -> {g}")
        if  g > maxG:
            maxG = g
            maxS = S
    return maxS

def Gtot(b, A) :
    return sum([pSousEnsemble(C, A, b.W, b.lam) * gainCliques(C, b.teta, b.G) for C in sousEnsembles(A)])

# probabilité que les cliques donne le sous-ensemble C
def pSousEnsemble(C, S, W, p_sat):
    p = 1
    for i, s in enumerate(S):
        if s in C:
            p *= pClique(i, S, W, p_sat)
        else:
            p *= 1 - pClique(i, S, W, p_sat)
    return p

# probabilité que l'élément i de S soit cliqué
def pClique(i, S, W, p_sat):
    pVu = 1 # proba que le i-ème élément de S soit vu
    for j in range(i):
        pVu *= (1-W[S[j]]*p_sat[j]) # proba que l'élément j+1 de S soit vu sachant le j-ème est vu
    return pVu*W[S[i]]

# gain d'un ensemble d'éléments cliqués C
def gainCliques(C, teta, G):
    sumTeta = sum([np.exp(t) for t in [teta[c] for c in C]])
    return sum([np.exp(teta[i])/sumTeta * G[i] for i in C])

# liste des sous-ensembles de l
def sousEnsembles(l):
    base = []
    lists = [base]
    for i in range(len(l)):
        orig = lists[:]
        new = l[i]
        for j in range(len(lists)):
            lists[j] = lists[j] + [new]
        lists = orig + lists

    return lists
