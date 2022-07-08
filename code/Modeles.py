#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module documentation goes here
   and here
   and ...

Created on Fri Jul 8 2022
@author: Armand
"""

from random import random
import numpy as np

class Modele:

    def __init__(self, n, k, lam, teta, W, G):
        # paramètre d'instance sous forme de liste
        self.n = n # nombre d'item
        self.k = k # nombre d'emplacement
        self.lam = lam # probabilité de satisfaction des emplacements
        self.teta = teta # utility des items
        self.W = W # attirance des items
        self.G = G # gain lié a chaque item

    def recommander(self, A):
        if len(A) != self.k:
            raise ValueError(f"taille de A ({len(A)}) différent de k ({self.k})")
        # cliques de l'utilisateur
        C = [0 for _ in A] # liste des cliques (1 pour un clique 0 sinon)
        i = 0
        stop = False
        while (not stop and i < len(A)) :
            a = A[i]
            # test attirance
            if random() < self.W[a]:
                C[i] = 1
                # test satisfaction
                if (random() < self.lam[i]):
                    stop = True
            i += 1
        # phase d'achat si il y a eu des cliques
        winner = -1
        if sum(C) > 0:
            lesTetas = [] # exp(teta) des items cliqués
            for i in range(len(A)):
                if C[i] == 1:
                    lesTetas.append(np.exp(self.teta[A[i]]))
            # achat
            p = random()
            i = 1
            while (p > sum(lesTetas[:i])/sum(lesTetas)):
                i += 1
            # winner
            j = 0
            while i > 0:
                if C[j] == 1:
                    i -= 1
                j += 1
            winner = A[j-1]
        return C, winner
