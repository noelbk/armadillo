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
  
  typedef typename   T1::elem_type  eT;
  typedef typename umat::elem_type ueT;
  
  const SpProxy<T1> P(in.m);
  
  arma_debug_check( (P.get_n_rows() != P.get_n_cols()), "symmatu()/symmatl(): given matrix must be square sized" );
  
  const bool upper = (in.aux_uword_a == 0);
  
  const uword n_nonzero = P.get_n_nonzero();
  
  if(n_nonzero == uword(0))
    {
    out.zeros(P.get_n_rows(), P.get_n_cols());
    return;
    }
  
  umat    out_locs(2, 2*n_nonzero);  // worse case scenario
  Col<eT> out_vals(   2*n_nonzero);
  
  ueT*  out_locs_ptr = out_locs.memptr();
   eT*  out_vals_ptr = out_vals.memptr();
  
  uword out_count = 0;
  
  typename SpProxy<T1>::const_iterator_type it = P.begin();
  
  if(upper)
    {
    // upper triangular: copy the diagonal and the elements above the diagonal
    
    for(uword in_count = 0; in_count < n_nonzero; ++in_count)
      {
      const uword row = it.row();
      const uword col = it.col();
      
      if(row < col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_locs_ptr[0] = col;
        out_locs_ptr[1] = row;
        out_locs_ptr += 2;
        
        const eT val = (*it);
        
        out_vals_ptr[0] = val;
        out_vals_ptr[1] = val;
        out_vals_ptr += 2;
        
        out_count += 2;
        }
      else
      if(row == col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_vals_ptr[0] = (*it);
        out_vals_ptr += 1;
        
        out_count++;
        }
      
      ++it;
      }
    }
  else
    {
    // lower triangular: copy the diagonal and the elements below the diagonal
    
    for(uword in_count = 0; in_count < n_nonzero; ++in_count)
      {
      const uword row = it.row();
      const uword col = it.col();
      
      if(row > col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_locs_ptr[0] = col;
        out_locs_ptr[1] = row;
        out_locs_ptr += 2;
        
        const eT val = (*it);
        
        out_vals_ptr[0] = val;
        out_vals_ptr[1] = val;
        out_vals_ptr += 2;
        
        out_count += 2;
        }
      else
      if(row == col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_vals_ptr[0] = (*it);
        out_vals_ptr += 1;
        
        out_count++;
        }
      
      ++it;
      }
    }
  
  const umat    tmp_locs(out_locs.memptr(), 2, out_count, false, false);
  const Col<eT> tmp_vals(out_vals.memptr(),    out_count, false, false);
  
  SpMat<eT> tmp(tmp_locs, tmp_vals, P.get_n_rows(), P.get_n_cols());
  
  out.steal_mem(tmp);
  }



template<typename T1>
inline
void
spop_symmat_cx::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_symmat_cx>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename   T1::elem_type  eT;
  typedef typename umat::elem_type ueT;
  
  const SpProxy<T1> P(in.m);
  
  arma_debug_check( (P.get_n_rows() != P.get_n_cols()), "symmatu()/symmatl(): given matrix must be square sized" );
  
  const bool upper   = (in.aux_uword_a == 0);
  const bool do_conj = (in.aux_uword_b == 1);
  
  const uword n_nonzero = P.get_n_nonzero();
  
  if(n_nonzero == uword(0))
    {
    out.zeros(P.get_n_rows(), P.get_n_cols());
    return;
    }
  
  umat    out_locs(2, 2*n_nonzero);  // worse case scenario
  Col<eT> out_vals(   2*n_nonzero);
  
  ueT*  out_locs_ptr = out_locs.memptr();
   eT*  out_vals_ptr = out_vals.memptr();
  
  uword out_count = 0;
  
  typename SpProxy<T1>::const_iterator_type it = P.begin();
  
  if(upper)
    {
    // upper triangular: copy the diagonal and the elements above the diagonal
    
    for(uword in_count = 0; in_count < n_nonzero; ++in_count)
      {
      const uword row = it.row();
      const uword col = it.col();
      
      if(row < col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_locs_ptr[0] = col;
        out_locs_ptr[1] = row;
        out_locs_ptr += 2;
        
        const eT val = (*it);
        
        out_vals_ptr[0] = val;
        out_vals_ptr[1] = (do_conj) ? std::conj(val) : val;
        out_vals_ptr += 2;
        
        out_count += 2;
        }
      else
      if(row == col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_vals_ptr[0] = (*it);
        out_vals_ptr += 1;
        
        out_count++;
        }
      
      ++it;
      }
    }
  else
    {
    // lower triangular: copy the diagonal and the elements below the diagonal
    
    for(uword in_count = 0; in_count < n_nonzero; ++in_count)
      {
      const uword row = it.row();
      const uword col = it.col();
      
      if(row > col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_locs_ptr[0] = col;
        out_locs_ptr[1] = row;
        out_locs_ptr += 2;
        
        const eT val = (*it);
        
        out_vals_ptr[0] = val;
        out_vals_ptr[1] = (do_conj) ? std::conj(val) : val;
        out_vals_ptr += 2;
        
        out_count += 2;
        }
      else
      if(row == col)
        {
        out_locs_ptr[0] = row;
        out_locs_ptr[1] = col;
        out_locs_ptr += 2;
        
        out_vals_ptr[0] = (*it);
        out_vals_ptr += 1;
        
        out_count++;
        }
      
      ++it;
      }
    }
  
  const umat    tmp_locs(out_locs.memptr(), 2, out_count, false, false);
  const Col<eT> tmp_vals(out_vals.memptr(),    out_count, false, false);
  
  SpMat<eT> tmp(tmp_locs, tmp_vals, P.get_n_rows(), P.get_n_cols());
  
  out.steal_mem(tmp);
  }



//! @}
