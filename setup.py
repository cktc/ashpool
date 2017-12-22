from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='ashpool',
    packages=['ashpool'],
    description='A library to compare data in dataframes.',
    url='http://github.com/cktc/ashpool',
    version='0.1',
    download_url='http://github.com/cktc/ashpool/archive/0.1.tar.gz',
    author='Christopher Cheung',
    author_email='chris.kt.cheung@gmail.com',
    license='MIT',
    install_requires=[
        'pandas', 'numpy'
    ],
    keywords=['compare pandas dataframes'], # arbitrary keywords
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        # 'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    zip_safe=False,
    )
