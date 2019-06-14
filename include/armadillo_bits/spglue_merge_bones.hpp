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



//! used only by SpSubview
class spglue_merge
  : public traits_glue_or
  {
  public:
  
  template<typename eT>
  arma_hot inline static void apply(SpMat<eT>& A, const uword A_sv_n_nonzero, const uword sv_row_start, const uword sv_row_end, const uword sv_col_start, const uword sv_col_end, const SpMat<eT>& B);
  };



//! @}
