import os
#import setuptools as se
from setuptools import Extension
from skbuild import setup

pkgs = {'gpt' : 'lib/gpt'}

subs = (
        'algorithms/iterative',
        'algorithms',
        'algorithms/approx',
        'core',
        'core/block',
        'core/io',
        'create',
        'qcd',
        'qcd/fermion',
        'qcd/fermion/preconditioner',
        'qcd/fermion/reference',
        'qcd/fermion/solver',
        'qcd/gauge',
)

for sub in subs:
    pkgs.update({'gpt.'+sub.replace('/','.'): 'lib/gpt/'+sub})

pkgs.update( {'cgpt' : 'lib/wrap'} )

setup(name='gpt',
    version='0.0.1',
    description='Python Measurements with Grid',
    package_dir=pkgs,
    packages=pkgs.keys())

# vim:expandtab:sw=4:sts=4
