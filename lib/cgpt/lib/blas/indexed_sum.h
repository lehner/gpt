/*
    GPT - Grid Python Toolkit
    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


template<typename dtype>
class cgpt_indexed_sum_job : public cgpt_blas_job_base {
 public:
  
  dtype* sp, *tp;
  int64_t *ip;
  std::vector<long> ss;
  long ts, id, stride, total, parallel, ortho;
  bool accumulate;

  cgpt_indexed_sum_job(void* _sp,void* _tp,void* _ip,
		       std::vector<long>& _ss,
		       long _ts, long _id, bool _accumulate) :
    sp((dtype*)_sp),
    tp((dtype*)_tp),
    ip((int64_t*)_ip),
    ss(_ss), ts(_ts), id(_id),
    accumulate(_accumulate) {

    stride = 1;
    for (long i=id;i<(long)ss.size();i++)
      stride *= ss[i];

    total=stride;
    for (long i=0;i<id;i++)
      total *= ss[i];

    parallel = (long)sqrt(total) + 1;
    ortho = (total + parallel - 1) / parallel;
  }
  
  std::string description() {
    std::ostringstream oss;
    oss << "IndexedSum(" << ss << "; " << parallel << " | " << ortho << ")";
    return oss.str();
  }

  virtual ~cgpt_indexed_sum_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.synchronise();
    
    deviceVector<dtype> tp_thread(parallel * ts);
    dtype* _tp_thread = &tp_thread[0];

    long _ts = ts;
    long _parallel = parallel;
    long _ortho = ortho;
    long _total = total;
    int64_t* _ip = ip;
    long _stride = stride;
    dtype* _sp = sp;
    dtype* _tp = tp;

    accelerator_for(p,_parallel,1, {
	for (long s=0;s<_ts;s++)
	  _tp_thread[s*_parallel + p] = 0;
	
	for (long o=0;o<_ortho;o++) {
	  long i = o*_parallel + p;
	  if (i < _total)
	    _tp_thread[_ip[i / _stride]*_parallel + p] += _sp[i];
	}
      });

    bool _accumulate=accumulate;
    accelerator_for(s,_ts,1,{
	dtype x = (_accumulate) ? _tp[s] : 0.0;
	for (long p=0;p<_parallel;p++)
	  x += _tp_thread[s*_parallel + p];
	_tp[s] = x;
      });
  }
};
