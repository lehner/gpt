import sys
sys.path.append('/cephfs/user/mabruno/gpt-mbruno46/lib')
import gpt
import numpy, os

from urllib import request

baseurl='https://github.com/waterret/Qlattice'
path='examples/propagators/sample-results/test-4nt8/results=1000/'

# download reference files for unit testing
for f in ['psrc-prop-0.field','pion-corr.txt']:
    if gpt.rank()==0:
        fname, header = request.urlretrieve(f'{baseurl}/raw/master/{path}/{f}', filename=f'./{f}')
    gpt.barrier()
    
prop = gpt.load('./psrc-prop-0.field')
gpt.message('Grid from qlat propagator =', prop.grid)

corr_pion=gpt.slice(gpt.trace(gpt.adj(prop)*prop),3)
corr_pion

with open('./pion-corr.txt','r') as f:
    txt=f.readlines()

# read lines corresponding to real part of time slices and 
# check difference w.r.t. what we have loaded above
for i in range(8):
    ref = float(txt[1+i*2].split(' ')[-1][:-1])
    diff = abs(ref - corr_pion[i].real)
    assert(diff < 1e-7) # propagator was computed in single precision
    gpt.message('Time slice %d difference %g' % (i,diff))

gpt.message('Test successful')

# remove reference files
for f in ['psrc-prop-0.field','pion-corr.txt']:
    if gpt.rank()==0:
        os.remove(f'./{f}')
