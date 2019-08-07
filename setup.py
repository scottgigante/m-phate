import os
import sys
from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'scikit-learn>=0.19.1',
    'pandas>=0.19.0,<0.24',
    'joblib',
    'scprep>=0.12.2',
    'phate>=0.4.3',
    'graphtools>=1.1'
]

test_requires = [
    'nose2',
    'coverage',
    'coveralls',
    'keras>=2.2',
    'tensorflow>=1.13',
    'parameterized',
]

doc_requires = [
    'sphinx<=1.8.5',
    'sphinxcontrib-napoleon',
    'autodocsumm',
    'ipykernel',
    'nbsphinx',
]

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >=3.5 required.")
elif sys.version_info[:2] < (3, 6):
    install_requires += ['matplotlib>=3.0,<3.1']
else:
    install_requires += ['matplotlib>=3.0']

version_py = os.path.join(os.path.dirname(
    __file__), 'm_phate', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

readme = open('README.md').read()

setup(name='m_phate',
      version=version,
      description='m-phate',
      author='Scott Gigante, Yale University',
      author_email='scott.gigante@yale.edu',
      packages=find_packages(),
      license='GNU General Public License Version 3',
      install_requires=install_requires,
      extras_require={'test': test_requires,
                      'doc': doc_requires},
      test_suite='nose2.collector.collector',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/scottgigante/m-phate',
      download_url="https://github.com/scottgigante/m-phate/archive/v{}.tar.gz".format(
          version),
      keywords=['big-data',
                'computational-biology',
                ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
      )
