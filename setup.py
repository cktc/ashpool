from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ashpool',
      version='0.1',
      description='Compare dataframes.',
      url='http://github.com/cktc/ashpool',
      author='Christopher Cheung',
      author_email='chris.kt.cheung@gmail.com',
      #   license='MIT',
      packages=['ashpool'],
      install_requires=[
          'pandas',
      ],
      zip_safe=False)
