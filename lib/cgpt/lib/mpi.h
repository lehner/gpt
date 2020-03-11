/*
  CGPT

  Authors: Christoph Lehner 2020
*/
EXPORT_BEGIN(global_rank) {
  return PyLong_FromLong(CartesianCommunicator::RankWorld());
} EXPORT_END();

