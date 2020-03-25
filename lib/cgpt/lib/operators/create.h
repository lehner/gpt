/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_fermion_operator(const std::string& optype, PyObject* args) {

  if (optype == "wilson_clover") {
    return cgpt_create_wilson_clover<vCoeff_t>(args);
  } else if (optype == "zmobius") {
    return cgpt_create_zmobius<vCoeff_t>(args);
  } else if (optype == "mobius") {
    return cgpt_create_mobius<vCoeff_t>(args);
  } else {
    ERR("Unknown operator type");
  }

}
