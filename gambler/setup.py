
   
from setuptools import setup

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().split('\n')

setup(
    name='adaptivesampling',
    version='1.0.0',
    author='Vasco Xu',
    email='vascoxu@uchicago.edu',
    description='Adaptive Sampling under Energy Budgets',
    packages=['adaptivesampling'],
    install_requires=reqs
)