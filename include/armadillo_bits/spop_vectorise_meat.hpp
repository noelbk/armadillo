// Copyright 2018 Ryan Curtin (http://ratml.org)
// Copyright 2018 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



//! \addtogroup spop_vectorise
//! @{



template<typename T1>
inline
void
spop_vectorise_col::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_col>& in)
  {
  arma_extra_debug_sigprint();
  
  spop_vectorise_col::apply_direct(out, in.m);
  }



template<typename T1>
inline
void
spop_vectorise_col::apply_direct(SpMat<typename T1::elem_type>& out, const T1& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  if(out.vec_state == 0)
    {
    out = expr;
    
    if( (out.n_elem == 0) || (out.n_nonzero == 0) )
      {
      out.zeros(out.n_elem, 1);
      }
    else
      {
      out.reshape(out.n_elem, 1);
      }
    }
  else
    {
    SpMat<eT> tmp = expr;
    
    if( (tmp.n_elem == 0) || (tmp.n_nonzero == 0) )
      {
      out.zeros(tmp.n_elem, 1);
      }
    else
      {
      tmp.reshape(tmp.n_elem, 1);
      
      out.steal_mem(tmp);
      }
    }
  }



template<typename T1>
inline
void
spop_vectorise_row::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_row>& in)
  {
  arma_extra_debug_sigprint();
  
  spop_vectorise_row::apply_direct(out, in.m);
  }



template<typename T1>
inline
void
spop_vectorise_row::apply_direct(SpMat<typename T1::elem_type>& out, const T1& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  if(out.vec_state == 0)
    {
    out = trans(expr);
    
    if( (out.n_elem == 0) || (out.n_nonzero == 0) )
      {
      out.zeros(1, out.n_elem);
      }
    else
      {
      out.reshape(1, out.n_elem);
      }
    }
  else
    {
    SpMat<eT> tmp = trans(expr);
    
    if( (tmp.n_elem == 0) || (tmp.n_nonzero == 0) )
      {
      out.zeros(1, tmp.n_elem);
      }
    else
      {
      tmp.reshape(1, tmp.n_elem);
      
      out.steal_mem(tmp);
      }
    }
  }



template<typename T1>
inline
void
spop_vectorise_all::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_vectorise_all>& in)
  {
  arma_extra_debug_sigprint();
  
  const uword dim = in.aux_uword_a;
  
  if(dim == 0)
    {
    spop_vectorise_col::apply_direct(out, in.m);
    }
  else
    {
    spop_vectorise_row::apply_direct(out, in.m);
    }
  }



//! @}
