/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_lattice_term {
private:
  ComplexD coef;
  cgpt_Lattice_base* lat;
  bool managed;
public:
  cgpt_lattice_term(ComplexD _coef, cgpt_Lattice_base* _lat, bool _managed) : coef(_coef), lat(_lat), managed(_managed) { };
  ~cgpt_lattice_term() { };
  void release() { if (managed)delete lat; };
  cgpt_Lattice_base* get_lat() { return lat; };
  ComplexD get_coef() { return coef; };
};  
