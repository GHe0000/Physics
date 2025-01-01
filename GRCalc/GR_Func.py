import sympy as sym
import numpy as np

# Gamma^a_bc


def CalcChristoffelSymbol(g, x):
    g_inv = g.inv()
    g_d = sym.derive_by_array(g, x)
    part_1 = np.einsum("il,jkl->ijk", g_inv, g_d, optimize="True")
    part_2 = np.einsum("il,klj->ijk", g_inv, g_d, optimize="True")
    part_3 = np.einsum("il,ljk->ijk", g_inv, g_d, optimize="True")
    ChristoffelSymbol = sym.simplify(sym.Rational(1, 2) *
                                     (part_1+part_2-part_3))
    return ChristoffelSymbol


def CalcRiemannTensor(g, x):
    Gamma = CalcChristoffelSymbol(g, x)
    part_1 = np.einsum('eca,dbe->abcd', Gamma, Gamma, optimize='optimal')
    part_2 = np.einsum('abcd->bacd', part_1)
    part_3 = np.einsum('adbc->abcd', sym.derive_by_array(Gamma, x))
    part_4 = np.einsum('abcd->bacd', part_3)
    RiemannTesnor = part_1 - part_2 - part_3 + part_4
    RiemannTesnor = sym.tensor.DenseNDimArray(RiemannTesnor)
    return RiemannTesnor

def CalcRicciTensor(g, x):
    RiemannTesnor = CalcRiemannTensor(g, x)
    RicciTensor = sym.simplify(sym.tensorcontraction(RiemannTesnor, (1, 3)))
    return RicciTensor


def CalcRicciScalar(g, x):
    g_inv = g.inv()
    RicciTensor = CalcRicciTensor(g, x)
    RicciScalar = np.einsum('ac,ac->', RicciTensor, g_inv, optimize='optimal')
    return RicciScalar
