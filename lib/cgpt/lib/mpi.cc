/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

EXPORT(global_rank,{
    return PyLong_FromLong(CartesianCommunicator::RankWorld());
  });

