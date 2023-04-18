from distutils.core import setup

from setuptools import find_packages


def get_readme():
    with open('README.md') as f:
        return f.read()


setup(name='model',
      version='1.0.0',
      description='Python Model related utilities',
      long_description=get_readme(),
      author='Brian Ball',
      url='https://github.com/brianjball/PythonUtilities/model',
      packages=find_packages()
      )
