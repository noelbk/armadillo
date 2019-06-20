// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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


//! \addtogroup spop_symmat
//! @{



template<typename T1>
inline
void
spop_symmat::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_symmat>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_spmat<T1> U(in.m);
  const SpMat<eT>& X   = U.M;
  
  arma_debug_check( (X.n_rows != X.n_cols), "symmatu()/symmatl(): given matrix must be square sized" );
  
  if(X.n_nonzero == uword(0))  { out.zeros(X.n_rows, X.n_cols); return; }
  
  const bool upper = (in.aux_uword_a == 0);
  
  const SpMat<eT> A = (upper) ? trimatu(X) : trimatl(X);  // in this case trimatu() and trimatl() return the same type
  const SpMat<eT> B = A.st();
  
  spop_symmat::merge_noalias(out, A, B);
  }



template<typename eT>
inline
void
spop_symmat::merge_noalias(SpMat<eT>& out, const SpMat<eT>& A, const SpMat<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  out.reserve(A.n_rows, A.n_cols, 2*A.n_nonzero); // worse case scenario
  
  typename SpMat<eT>::const_iterator x_it  = A.begin();
  typename SpMat<eT>::const_iterator x_end = A.end();
  
  typename SpMat<eT>::const_iterator y_it  = B.begin();
  typename SpMat<eT>::const_iterator y_end = B.end();
  
  uword count = 0;
  
  while( (x_it != x_end) || (y_it != y_end) )
    {
    eT out_val;
    
    const uword x_it_col = x_it.col();
    const uword x_it_row = x_it.row();
    
    const uword y_it_col = y_it.col();
    const uword y_it_row = y_it.row();
    
    bool use_y_loc = false;
    
    if(x_it == y_it)
      {
      // this can only happen on the diagonal
      
      out_val = (*x_it);
      
      ++x_it;
      ++y_it;
      }
    else
      {
      if((x_it_col < y_it_col) || ((x_it_col == y_it_col) && (x_it_row < y_it_row))) // if y is closer to the end
        {
        out_val = (*x_it);
        
        ++x_it;
        }
      else
        {
        out_val = (*y_it);
        
        ++y_it;
        
        use_y_loc = true;
        }
      }
    
    access::rw(out.values[count]) = out_val;
    
    const uword out_row = (use_y_loc == false) ? x_it_row : y_it_row;
    const uword out_col = (use_y_loc == false) ? x_it_col : y_it_col;
    
    access::rw(out.row_indices[count]) = out_row;
    access::rw(out.col_ptrs[out_col + 1])++;
    ++count;
    }
  
  const uword out_n_cols = out.n_cols;
  
  uword* col_ptrs = access::rwp(out.col_ptrs);
  
  // Fix column pointers to be cumulative.
  for(uword c = 1; c <= out_n_cols; ++c)
    {
    col_ptrs[c] += col_ptrs[c - 1];
    }
  
  // quick resize without reallocating memory and copying data
  access::rw(         out.n_nonzero) = count;
  access::rw(     out.values[count]) = eT(0);
  access::rw(out.row_indices[count]) = uword(0);
  }



template<typename T1>
inline
void
spop_symmat_cx::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_symmat_cx>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_spmat<T1> U(in.m);
  const SpMat<eT>& X   = U.M;
  
  arma_debug_check( (X.n_rows != X.n_cols), "symmatu()/symmatl(): given matrix must be square sized" );
  
  if(X.n_nonzero == uword(0))  { out.zeros(X.n_rows, X.n_cols); return; }
  
  const bool upper   = (in.aux_uword_a == 0);
  const bool do_conj = (in.aux_uword_b == 1);
  
  const SpMat<eT> A = (upper) ? trimatu(X) : trimatl(X);  // in this case trimatu() and trimatl() return the same type
  
  if(do_conj)
    {
    const SpMat<eT> B = A.t();
    
    spop_symmat::merge_noalias(out, A, B);
    }
  else
    {
    const SpMat<eT> B = A.st();
    
    spop_symmat::merge_noalias(out, A, B);
    }
  }



//! @}
