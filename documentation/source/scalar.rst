Example: Scalar Field Theory
****************************

.. code:: ipython3

    import gpt as g
    
    m0crit=4.0
    m0=g.default.get_float("--mass",0.1) + m0crit
    
    grid=g.grid(g.default.grid, g.default.precision)
    
    src=g.complex(grid)
    src[:]=0
    src[0,0,0,0]=1
    
    # Create a free Klein-Gordon operator (spectrum from mass^2-16 .. mass^2)
    def A(dst,src,mass):
        assert(dst != src)
        dst @= (mass**2.)*src
        for i in range(4):
            dst += g.cshift(src, i, 1) + g.cshift(src, i, -1) - 2*src
    
    # find largest eigenvalue
    powit=g.algorithms.iterative.power_iteration({"eps":1e-6,"maxiter":100})
    g.message("Largest eigenvalue: ", powit(lambda i,o: A(o,i,m0),src)[0])
    
    # perform CG
    psi=g.lattice(src)
    psi[:]=0
    
    cg=g.algorithms.iterative.cg({"eps":1e-8,"maxiter":1000})
    cg(lambda i,o: A(o,i,m0),src,psi)
    
    g.meminfo()
    
    # Test CG
    tmp=g.lattice(psi)
    A(tmp,psi,m0)
    g.message("True residuum:", g.norm2( tmp - src ))


.. parsed-literal::

    GPT :       0.408669 s : eval_max[ 0 ] = 9.2529
    GPT :       0.440208 s : eval_max[ 1 ] = 10.7673
    GPT :       0.463457 s : eval_max[ 2 ] = 11.8817
    GPT :       0.472082 s : eval_max[ 3 ] = 12.7639
    GPT :       0.540482 s : eval_max[ 4 ] = 13.4826
    GPT :       0.550098 s : eval_max[ 5 ] = 14.0762
    GPT :       0.551125 s : eval_max[ 6 ] = 14.5694
    GPT :       0.560096 s : eval_max[ 7 ] = 14.9799
    GPT :       0.570398 s : eval_max[ 8 ] = 15.3213
    GPT :       0.583492 s : eval_max[ 9 ] = 15.6044
    GPT :       0.590788 s : eval_max[ 10 ] = 15.8381
    GPT :       0.616045 s : eval_max[ 11 ] = 16.03
    GPT :       0.632026 s : eval_max[ 12 ] = 16.1867
    GPT :       0.635078 s : eval_max[ 13 ] = 16.314
    GPT :       0.665744 s : eval_max[ 14 ] = 16.4167
    GPT :       0.682065 s : eval_max[ 15 ] = 16.4992
    GPT :       0.684466 s : eval_max[ 16 ] = 16.5652
    GPT :       0.698942 s : eval_max[ 17 ] = 16.6176
    GPT :       0.701244 s : eval_max[ 18 ] = 16.6591
    GPT :       0.707136 s : eval_max[ 19 ] = 16.6919
    GPT :       0.716783 s : eval_max[ 20 ] = 16.7177
    GPT :       0.719413 s : eval_max[ 21 ] = 16.738
    GPT :       0.726697 s : eval_max[ 22 ] = 16.7538
    GPT :       0.730942 s : eval_max[ 23 ] = 16.7663
    GPT :       0.756783 s : eval_max[ 24 ] = 16.776
    GPT :       0.766245 s : eval_max[ 25 ] = 16.7835
    GPT :       0.776332 s : eval_max[ 26 ] = 16.7894
    GPT :       0.782739 s : eval_max[ 27 ] = 16.794
    GPT :       0.784998 s : eval_max[ 28 ] = 16.7976
    GPT :       0.791529 s : eval_max[ 29 ] = 16.8003
    GPT :       0.800217 s : eval_max[ 30 ] = 16.8025
    GPT :       0.802696 s : eval_max[ 31 ] = 16.8042
    GPT :       0.806765 s : eval_max[ 32 ] = 16.8055
    GPT :       0.818400 s : eval_max[ 33 ] = 16.8065
    GPT :       0.822189 s : eval_max[ 34 ] = 16.8073
    GPT :       0.834184 s : eval_max[ 35 ] = 16.8079
    GPT :       0.854798 s : eval_max[ 36 ] = 16.8084
    GPT :       0.873497 s : eval_max[ 37 ] = 16.8087
    GPT :       0.877957 s : eval_max[ 38 ] = 16.809
    GPT :       0.890104 s : eval_max[ 39 ] = 16.8092
    GPT :       0.897069 s : eval_max[ 40 ] = 16.8094
    GPT :       0.900714 s : eval_max[ 41 ] = 16.8095
    GPT :       0.929664 s : eval_max[ 42 ] = 16.8096
    GPT :       0.936316 s : eval_max[ 43 ] = 16.8097
    GPT :       0.948886 s : eval_max[ 44 ] = 16.8098
    GPT :       0.951324 s : eval_max[ 45 ] = 16.8098
    GPT :       0.957485 s : eval_max[ 46 ] = 16.8099
    GPT :       0.969620 s : eval_max[ 47 ] = 16.8099
    GPT :       1.001650 s : eval_max[ 48 ] = 16.8099
    GPT :       1.022748 s : eval_max[ 49 ] = 16.8099
    GPT :       1.073016 s : eval_max[ 50 ] = 16.81
    GPT :       1.077150 s : Converged
    GPT :       1.079259 s : Largest eigenvalue:  16.809952552893705
    GPT :       1.085259 s : res^2[ 1 ] = 0.103071
    GPT :       1.088590 s : res^2[ 2 ] = 0.0231099
    GPT :       1.099775 s : res^2[ 3 ] = 0.00839726
    GPT :       1.122351 s : res^2[ 4 ] = 0.00429588
    GPT :       1.133646 s : res^2[ 5 ] = 0.00272869
    GPT :       1.152022 s : res^2[ 6 ] = 0.00178488
    GPT :       1.163438 s : res^2[ 7 ] = 0.000863748
    GPT :       1.166113 s : res^2[ 8 ] = 0.000179348
    GPT :       1.168090 s : res^2[ 9 ] = 1.58948e-33
    GPT :       1.168767 s : Converged in 0.0846589 s
    GPT :       1.169415 s : ==========================================================================================
    GPT :       1.170030 s :                                  GPT Memory Report                
    GPT :       1.171864 s : ==========================================================================================
    GPT :       1.172928 s :  Index    Grid                           Precision    OType           CBType       Size/GB          Created at time     
    GPT :       1.174797 s :  0        [4, 4, 4, 4]                   double       ot_complex      full         3.814697265625e-06 0.407694 s          
    GPT :       1.175496 s :  1        [4, 4, 4, 4]                   double       ot_complex      full         3.814697265625e-06 1.082682 s          
    GPT :       1.176056 s : ==========================================================================================
    GPT :       1.176479 s :    Total: 7.62939e-06 GB 
    GPT :       1.176899 s : ==========================================================================================
    GPT :       1.232319 s : True residuum: 2.4471648282424126e-31

