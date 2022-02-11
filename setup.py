# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Install kfac."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='kfac',
    version='0.2.4',
    description='K-FAC for TensorFlow',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/kfac',
    license='Apache 2.0',
    packages=find_packages(exclude=[
        'kfac.examples.*',
        'kfac.python.kernel_tests.*',
    ]),
    install_requires=[
        'numpy',
        'six',
        'tensorflow-probability==0.8',
        'h5py<3',
    ],
    extras_require={
        # It's possible that you might need to put tensorflow<2.0 here:
        'tensorflow': ['tensorflow>=1.14'],
        # It's possible that you might need to put tensorflow-gpu<2.0 here:
        'tensorflow_gpu': ['tensorflow-gpu>=1.14'],
        # dm-sonnet<2.0 will force tensorflow<2.0 in the tests:
        'tests': ['pytest', 'dm-sonnet<2.0', 'numpy<1.20'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning',
)
