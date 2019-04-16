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



//! \addtogroup fn_shuffle
//! @{


template<typename T1>
arma_warn_unused
arma_inline
typename
enable_if2
  <
  is_arma_type<T1>::value && resolves_to_vector<T1>::yes,
  const Op<T1, op_shuffle_vec>
  >::result
shuffle
  (
  const T1& X
  )
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_shuffle_vec>(X);
  }



template<typename T1>
arma_warn_unused
arma_inline
typename
enable_if2
  <
  is_arma_type<T1>::value && resolves_to_vector<T1>::no,
  const Op<T1, op_shuffle>
  >::result
shuffle
  (
  const T1& X
  )
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_shuffle>(X);
  }



template<typename T1>
arma_warn_unused
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value),
  const Op<T1, op_shuffle>
  >::result
shuffle
  (
  const T1&   X,
  const uword dim
  )
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_shuffle>(X, dim, 0);
  }



//! @}
