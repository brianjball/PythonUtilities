from distutils.core import setup

from setuptools import find_packages


def get_readme():
    with open('README.md') as f:
        return f.read()


setup(name='mlflow',
      version='1.0.0',
      description='Python MLFlow related utilities',
      long_description=get_readme(),
      author='Brian Ball',
      url='https://github.com/brianjball/PythonUtilities/mlflow',
      packages=find_packages()
      )
