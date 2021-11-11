from setuptools import setup

setup(
    name='ejemplospythonaq',
    version='',
    packages=[''],
    url='',
    license='',
    author='aq',
    author_email='juaquiro@gmail.com',
    description='Ejemplos Python AQ',

    # if version is necessary specify it as 'numpy==1.19.2' for example
    # for checking requirements (and use them for generate requirements.txt file) run $ pip freeze
    install_requires=['numpy==1.21.4', 'opencv-python==4.5.4.58',
                      'matplotlib==3.4.3', 'scikit-image==0.18.3',
                      'jupyter==1.0.0', 'PyQt5==5.15.6', 'ipynb==0.5.1']
)
