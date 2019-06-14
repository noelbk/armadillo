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


//! \addtogroup spglue_merge
//! @{



template<typename eT>
arma_hot
inline
void
spglue_merge::apply(SpMat<eT>& A, const uword A_sv_n_nonzero, const uword sv_row_start, const uword sv_row_end, const uword sv_col_start, const uword sv_col_end, const SpMat<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(A.n_rows, A.n_cols, B.n_rows, B.n_cols, "merge");
  
  const uword merge_n_nonzero = A.n_nonzero - A_sv_n_nonzero + B.n_nonzero;
  
  if(merge_n_nonzero == 0)  { A.zeros(); return; }
  
  if(A_sv_n_nonzero == A.n_nonzero)
    {
    // A has all of its elements in the subview
    // so the merge is equivalent to overwrite of A
    
    A = B;
    return;
    }
  
  if(A_sv_n_nonzero > (A.n_nonzero/2))
    {
    // A has most of its elements in the subview,
    // so regenerate matrix A with zeros in the subview region
    // in order to increase merging efficiency
    
    SpMat<eT> tmp(arma_reserve_indicator(), A.n_rows, A.n_cols, A.n_nonzero - A_sv_n_nonzero);
    
    typename SpMat<eT>::const_iterator A_it     = A.begin();
    typename SpMat<eT>::const_iterator A_it_end = A.end();
    
    uword tmp_count = 0;
    
    for(; A_it != A_it_end; ++A_it)
      {
      const uword A_it_row = A_it.row();
      const uword A_it_col = A_it.col();
      
      const bool inside_box = ((A_it_row >= sv_row_start) && (A_it_row <= sv_row_end)) && ((A_it_col >= sv_col_start) && (A_it_col <= sv_col_end));
      
      if(inside_box == false)
        {
        access::rw(tmp.values[tmp_count])      = (*A_it);
        access::rw(tmp.row_indices[tmp_count]) = A_it_row;
        access::rw(tmp.col_ptrs[A_it_col + 1])++;
        ++tmp_count;
        }
      }
    
    for(uword i=0; i < tmp.n_cols; ++i)
      {
      access::rw(tmp.col_ptrs[i + 1]) += tmp.col_ptrs[i];
      }
    
    A.steal_mem(tmp);
    }
  
  
  SpMat<eT> out(arma_reserve_indicator(), A.n_rows, A.n_cols, merge_n_nonzero);
  
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
      out_val = (*y_it);
      
      ++x_it;
      ++y_it;
      }
    else
      {
      if((x_it_col < y_it_col) || ((x_it_col == y_it_col) && (x_it_row < y_it_row))) // if y is closer to the end
        {
        const bool x_inside_box = ((x_it_row >= sv_row_start) && (x_it_row <= sv_row_end)) && ((x_it_col >= sv_col_start) && (x_it_col <= sv_col_end));
        
        out_val = (x_inside_box) ? eT(0) : (*x_it);
        
        ++x_it;
        }
      else
        {
        out_val = (*y_it);
        
        ++y_it;
        
        use_y_loc = true;
        }
      }
    
    if(out_val != eT(0))
      {
      access::rw(out.values[count]) = out_val;
      
      const uword out_row = (use_y_loc == false) ? x_it_row : y_it_row;
      const uword out_col = (use_y_loc == false) ? x_it_col : y_it_col;
      
      access::rw(out.row_indices[count]) = out_row;
      access::rw(out.col_ptrs[out_col + 1])++;
      ++count;
      }
    }
  
  arma_check( (count != merge_n_nonzero), "spglue_merge::apply(): internal error: count != merge_n_nonzero" );
  
  const uword out_n_cols = out.n_cols;
  
  uword* col_ptrs = access::rwp(out.col_ptrs);
  
  for(uword c = 1; c <= out_n_cols; ++c)
    {
    col_ptrs[c] += col_ptrs[c - 1];
    }
  
  A.steal_mem(out);
  }



//! @}
