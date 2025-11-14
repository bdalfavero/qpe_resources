"""This file comes from
https://github.com/Shashank-G-M/Perturbative_Trotter_Error/blob/main/pert_trotter/error_pert.py
See the paper arXiv:2312.13282"""

import numpy as np
import scipy as sp
from itertools import accumulate
from openfermion import commutator
from copy import copy
#Add additional imports related to cupy if you are using GPU







def get_V1_sarray(fragments_list):
  '''
  Build sparse matrix corresponding to Trotter perturbation correction opeartor V1 using the list of fragments.

  Args:
      framents_list (list): List of fragments of the Hamiltonian as scipy.sparse.csc_matrix objects.

  Returns:
      scipy.sparse.csc_matrix: Sparse matrix corresponding to V1.
  '''
  acc_frags = list(accumulate(fragments_list))
  V1 = 0
  for i in range (1, len(fragments_list)):
    V1 += commutator(acc_frags[i-1], fragments_list[i])/2
  return V1





#Get contribution due to V1 term
def get_V1_contri(v, w, CEDA_num_only = False, V1_sarray = None, fragments_list = None):
  '''
  Given a list of fragments of the Hamiltonian along with its eigenvalues and eigenvectors, returns the
  second order perturbation contribution due to V1 operator for the unperturbed eigenvalue v[0].

  Args:
      v (np.array): List of eigenvalues of the Hamiltonian
      w (np.array): Array of corresponding eigenvectors of the Hamiltonian
      CEDA_num_only (bool, optional): If True, only the numerator of CEDA contribution is returned along with V1 contribution. Defaults to False.
      V1_sarray (csc_matrix, optional): Sparse matrix corresponding to V1. Defaults to None.
      fragments_list (list, optional): List of fragments of the Hamiltonian as scipy.sparse.csc_matrix objects. Defaults to None.

  Returns:
      if CEDA_num_only == False --> (float, (float, float)): (V1 contribution, (CEDA upper bound, CEDA lower bound))
                          else  --> (float, float): (V1 contribution, CEDA numerator)
  '''
  v0 = v[0]
  if fragments_list == None and V1_sarray == None:
    raise ValueError('Either fragments_list or V1_sarray must be provided')
  if V1_sarray == None:
    get_V1_sarray(fragments_list)
  else:
    V1 = V1_sarray
  w0_sparse = sp.sparse.csc_matrix(w[:,[0]])
  V1w0 = V1*w0_sparse
  contri = 0
  for i in range (1,len(v)):
    num = sp.sparse.csc_matrix(w[:,[i]].T)*V1w0
    contri += np.abs(num.toarray())**2/(v0-v[i])
  CEDA_num = (V1w0.T*V1w0).toarray()[0,0]
  if CEDA_num_only == True:
    return(contri[0,0], CEDA_num)
  else:
    return (contri[0,0], (CEDA_num/(v[0]-v[-1]), CEDA_num/(v[0]-v[1])))








def efficient_V1_contri(fragments_list_sarray, v, w, track = True):
  '''
  Given a list of fragments of the Hamiltonian along with eigenvalues and eigenvectors, returns the
  second order perturbation contribution due to V1 operator for the unperturbed eigenvalue v[0]. This
  routine does not rely on building V1 operator and hence is more efficient.

  Args:
      fragments_list_sarray (list): List or Hamiltonian fragments as scipy csc_matrix objects
      v: List of eigenvalues of the Hamiltonian
      w: Array of corresponding eigenvectors of the Hamiltonian. Could be approximate states as well.
      track (bool, optional): If True, prints progress. Defaults to True.

  Returns:
      float: V1 contribution.

  '''
  if sp.sparse.issparse(fragments_list_sarray[0]) != True:
    raise ValueError('Fragments must be sparse')
  if sp.sparse.issparse(w) == True:
    raise ValueError('Eigenvectors must be dense')

  frags_len = len(fragments_list_sarray)
  if track == True:
    print ('Total number of loops expected: ', frags_len-1)

  w0 = sp.sparse.csc_matrix(w[:,[0]])
  w0T = w0.T
  wT = w.T

  V1_conri = 0
  tracker = 0
  Hmu = fragments_list_sarray[0]
  Hmu_nu_w0 = 0
  w0_Hmu_nu = 0
  for i in range (1, frags_len):
    Hnu = fragments_list_sarray[i]
    Hnu_w0 = Hnu*w0
    Hmu_nu_w0 = Hmu_nu_w0 + Hmu*Hnu_w0

    w0_Hmu = w0T*Hmu
    w0_Hmu_nu = w0_Hmu_nu + w0_Hmu*Hnu
    Hmu += Hnu

    if track == True:
      if i/(frags_len-1)*100 >=  tracker:
        print ('\rProgress: [%d%%]'%int(tracker), end="")
        tracker += 5

  print ('\nCollecting the results...')
  V1_num_1 = (wT@Hmu_nu_w0.toarray()).T
  V1_num_2 = w0_Hmu_nu.toarray()@w
  V1_num = V1_num_1 - V1_num_2
  V1_num = 0.25*np.abs(V1_num[0,:])**2

  V1_contri = sum([V1_num[i]/(v[0] - v[i]) for i in range (1, len(v))])

  return V1_contri








def get_v2_sarray(fragments_list, dim = None):
  '''
  Build sparse matrix corresponding to Trotter perturbation correction opeartor V2 using the list of fragments.

  Args:
      fragments_list (list): List of fragments of the Hamiltonian as scipy.sparse.csc_matrix objects.
      dim (int, optional): Dimension of the hamiltonian fragments. Defaults to None.

  Returns:
      float: scipy csc_matirx representing v2 operator.
  '''
  if dim == None:
    dim = fragments_list[-1].shape[0]
  frags_len = len(fragments_list)
  frag_sums_l2r = list(accumulate(fragments_list))
  temp = reversed(fragments_list)
  frag_sums_r2l = list(accumulate(temp))
  frag_sums_r2l = list(reversed(frag_sums_r2l))
  frag_sums_r2l.append(sp.sparse.csc_matrix((dim, dim), dtype='complex128'))
  frag_combs_V1_v2 = [(frag_sums_l2r[i-1], fragments_list[i], frag_sums_r2l[i+1]) for i in range (1, frags_len)]
  v2 = 0
  for i,j,k in frag_combs_V1_v2:
    V1_term = commutator(i, j)
    v2 += - commutator(V1_term, k)*1/3 - commutator(V1_term, j)*1/6
  return v2









#Get contribution due to v2 term
def get_v2_contri(w0_sparse, dim = None, v2_sarray = None, fragments_list = None):
  '''
  Given scipy csc_matrix of (apprximate) ground state as column vector, and either the v2_sarray or list of fragments,
  find the contirbution to perturbation series coming from the v2 operator.

  Args:
      w0_sparse (scipy.sparse.csc_matrix): Column vector with respect to which v2 contribution is to be evaluated.
                                           For example, it could be the exact ground state, or the CISD ground state.
                                           But the w0_sparse.shape[0] = fragments_list[0].shape[1].
      dim (int, optional): Dimension of the hamiltonian fragments. Defaults to None.
      v2_sarray (scipy.sparse.csc_matrix, optional): Sparse matrix corresponding to v2. Defaults to None.
      fragments_list (list, optional): List of fragments of the Hamiltonian as scipy.sparse.csc_matrix objects. Defaults to None.

  Returns:
      float: v2 contribution.
  '''
  if (fragments_list == None) and (sp.sparse.issparse(v2_sarray) != True):
    raise ValueError('Either fragments_list or v2_sarray must be provided')
  if sp.sparse.issparse(v2_sarray) != True:
    get_v2_sarray(fragments_list, dim = dim)
  else:
    v2 = v2_sarray
  contri = w0_sparse.T*v2*w0_sparse

  return contri.toarray()[0,0]












def efficient_v2_contri(fragments_list_sarray, w0_sparse, track = True):
  '''
  Given a list of fragments of the Hamiltonian along with (approx) ground state, returns the
  perturbation correction due to v2. The key ingerident for the efficiency is acting sparse
  arrays on vectors and then manipulting the vectors instead of multiplying the arrays and then acting on vectors.

  Args:
      fragments_list_sarray: A list or array of fragments of the Hamiltonian as scipy.sparse.csc_matrix arrays

      w0_sparse (scipy.sparse.csc_mateix): A column vector with respect to which v2 contribution is to be evaluated. For example,
                                       it could be the exact ground state, or the CISD ground state.
      track (bool, optional): If True, prints progress. Defaults to True.

  Returns:
      float: v2 contribution.
  '''
  frags_on_w0 = [frag*w0_sparse for frag in fragments_list_sarray]
  rks_w0 = list(reversed(list(accumulate(reversed(frags_on_w0)))))              #Can be made more efficient by finding the elements lazily instead of materializing the whole list at once.
  rks_w0.append(0)                                                              #Needed to avoid list index out of range error in the following for loop.

  frags_len = len(fragments_list_sarray)
  if track == True:
    print ('Total number of loops: ', frags_len-1)

  tracker = 10
  v2_contri = 0
  ri = 0

  for i in range (1, frags_len):
    ri = ri + fragments_list_sarray[i-1].tocsr()
    rj = fragments_list_sarray[i].tocsr()

    w0_ri = w0_sparse.T*ri
    w0_rirj = w0_ri*rj
    w0_rj = w0_sparse.T*rj
    w0_rirjT = w0_rj*ri
    w0_V1_term = w0_rirj - w0_rirjT

    rk_w0 = rks_w0[i+1]/3
    rj_w0 = rj*w0_sparse/6
    rkj_w0 = rk_w0 + rj_w0

    v2_contri += -2*(w0_V1_term*rkj_w0).toarray()[0,0]

    if track == True:
      if i/(frags_len-1)*100 >= tracker:
        print ('\rProgress: [%d%%]'%int(tracker), end="")
        tracker += 5
  print ('\n')
  return v2_contri












def efficient_v2_contri_2_order_Trotter(fragments_list_sarray, w0_sparse, track = True):
  '''
  Given a list of fragments of the Hamiltonian along with (approx) ground state, returns the
  perturbation correction due to v2 of the second order Trotter product formual. The key ingerident for the efficiency is acting sparse
  arrays on vectors and then manipulting the vectors instead of multiplying the arrays and then acting on vectors.

  Args:
      fragments_list_sarray: A list or array of fragments of the Hamiltonian as scipy.sparse.csc_matrix arrays.

      w0_sparse (scipy.sparse.csc_mateix): A column vector with respect to which v2 contribution is to be evaluated. For example,
                                       it could be the exact ground state, or the CISD ground state.
      track (bool, optional): If True, prints progress. Defaults to True.

  Returns:
      float: v2 contribution to second order Trotter.
  '''
  frags_on_w0 = [frag*w0_sparse for frag in fragments_list_sarray]
  frags_on_w0 = frags_on_w0 + frags_on_w0[::-1]
  rks_w0 = list(reversed(list(accumulate(reversed(frags_on_w0)))))              #Can be made more efficient by finding the elements lazily instead of materializing the whole list at once.
  rks_w0.append(0)                                                              #Needed to avoid list index out of range error in the following for loop.

  frags_len = len(fragments_list_sarray)
  if track == True:
    print ('Total number of loops: ', 2*frags_len-1)

  tracker = 10
  v2_contri = 0
  ri = 0

  map_idx = lambda i: i if i < frags_len else 2*frags_len-i-1

  for i in range (1, 2*frags_len):
    ri = ri + fragments_list_sarray[map_idx(i-1)].tocsr()
    rj = fragments_list_sarray[map_idx(i)].tocsr()

    w0_ri = w0_sparse.T*ri
    w0_rirj = w0_ri*rj
    w0_rj = w0_sparse.T*rj
    w0_rirjT = w0_rj*ri
    w0_V1_term = w0_rirj - w0_rirjT

    rk_w0 = rks_w0[i+1]/3
    rj_w0 = rj*w0_sparse/6
    rkj_w0 = rk_w0 + rj_w0

    v2_contri += -2*(w0_V1_term*rkj_w0).toarray()[0,0]

    if track == True:
      if i/(2*frags_len-1)*100 >= tracker:
        print ('\rProgress: [%d%%]'%int(tracker), end="")
        tracker += 5

  v2_contri = v2_contri/8                                     #Because of t/2 coming from 2nd order Trotter

  print ('\n')
  return v2_contri











def Trotter_alpha_error(fragments_list_sarray, gpu = False):
  '''
  Obtain the operator norm error alpha.

  Args:
      fragments_list_sarray (scipy.sparse.csc_matrix): List of Hamiltonian fragments as scipy sparse matrices.
      gpu (bool, optional): If True, uses python package cupy to perform calculations on GPU. Defaults to False.

  Returns:
      float: Operator norm error alpha.
  '''
  frags_len = len(fragments_list_sarray)
  alpha = 0
  for i in range (frags_len-1):
    frag12 = fragments_list_sarray[i]*sum(fragments_list_sarray[i+1:])
    term = frag12 - frag12.T
    if gpu == False:
      alpha += 2*np.abs(sp.sparse.linalg.eigsh(1.j*term, k = 1, which='LM', tol = 1e-5, return_eigenvectors=False)[0])
      # print (alpha)
    else:
      term = cusparse.csc_matrix(term)
      alpha += 2*np.abs(gpu_eigsh(1.j*term, k=1, which='LM', tol=1e-5, maxiter = 5000, return_eigenvectors=False)[0])
      # print (alpha)
    print ('\rProgress: [%d%%]'%int(100*(i+1)/(frags_len-1)), end="")
  return alpha











def Trotter_2nd_order_alpha_error(fragments_list_sarray, gpu = False, tol = None):
  '''
  Obtain the operator norm error alpha for the second order Trotter, based on preposition 10 of "Theory of Trotter error with commutator scaling".

  Args:
      fragments_list_sarray (scipy.sparse.csc_matrix): List of Hamiltonian fragments as scipy sparse matrices.
      gpu (bool, optional): If True, uses python package cupy to perform calculations on GPU. Defaults to False.
      tol (float, optional): It was observed that the value of alpha converges very fast and hence the subsequent loops can be ignored. Defaults to None.

  Returns:
      float: Operator norm error alpha.
  '''
  frags_len = len(fragments_list_sarray)
  alpha = 0
  Si = sum(fragments_list_sarray)
  for i in range (frags_len - 1):
    old_alpha = copy(alpha)

    Hi = fragments_list_sarray[i]
    Si = Si - Hi
    SiHi = Si*Hi
    SiHiSi = SiHi*Si
    SiSiHi = Si*SiHi
    term1 = SiSiHi -2*SiHiSi + SiSiHi.T
    del SiHiSi, SiSiHi
    if gpu == True:
      cp._default_memory_pool.free_all_blocks()

    HiSiHi = Hi*SiHi
    SiHiHi = SiHi*Hi
    term2 = SiHiHi.T -2*HiSiHi + SiHiHi
    del SiHiHi, HiSiHi, SiHi
    if gpu == True:
      cp._default_memory_pool.free_all_blocks()

    if gpu == True:
      term1 = term1.get()
      term2 = term2.get()

    alpha += np.abs(sp.sparse.linalg.eigsh(term1, k = 1, which='LM', tol = 1e-5, return_eigenvectors=False)[0])/12
    alpha += np.abs(sp.sparse.linalg.eigsh(term2, k = 1, which='LM', tol = 1e-5, return_eigenvectors=False)[0])/24

    del term1, term2
    if gpu == True:
      cp._default_memory_pool.free_all_blocks()

    if tol != None:
      if (np.abs(alpha - old_alpha) < tol) and (i > 5):
        break

    print ('\rProgress: [%d%%]'%int(100*(i+1)/(frags_len-1)), end="")
  print ('\n')
  return alpha




