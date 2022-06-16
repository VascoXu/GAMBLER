from setuptools import setup

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().split('\n')

setup(
    name='gambler',
    version='1.0.0',
    author='Vasco Xu',
    email='vascoxu@uchicago.edu',
    description='Adaptive sampling under energy budgets.',
    packages=['gambler'],
    install_requires=reqs
)
