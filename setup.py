'''
setup.py for ConvLab-2
'''
import sys
import os
from typing import List
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class LibTest(TestCommand):

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        ret = os.system("pytest --cov=ConvLab-2 tests/ --cov-report term-missing")
        sys.exit(ret >> 8)


def read_requirements(require_file: str = './requirements.txt') -> List[str]:
    """read the dependency requirements from the file

    Returns:
        List[str]: the list of dependency packages
    """
    if not os.path.exists(require_file):
        raise FileNotFoundError(f'{require_file} file not found')
    with open(require_file, 'r+') as f:
        return list(f.readlines())

setup(
    name='ConvLab-2',
    version='0.0.1',
    packages=find_packages(exclude=[]),
    license='Apache',
    description='Task-oriented Dialog System Toolkits',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
                'Development Status :: 2 - Pre-Alpha',
                'License :: OSI Approved :: Apache Software License',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=read_requirements(),
    extras_require={
        'develop': read_requirements('./requirements-dev.txt')
    },
    cmdclass={'test': LibTest},
    entry_points={
        'console_scripts': [
            "ConvLab-2-report=convlab2.scripts:report"
        ]
    },
    include_package_data=True,
    url='https://github.com/thu-coai/ConvLab-2',
    author='thu-coai',
    author_email='thu-coai-developer@googlegroups.com',
    python_requires='>=3.5',
    zip_safe=False
)
