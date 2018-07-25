
from setuptools import setup
setup(name='ppg_features',
      version='1.0',
      description='Filter and extract features from finger PPG',
      long_description='',
      author='Jaron C Whittington',
      author_email='jaronwhitty@gmail.com',
      url='https://github.com/JaronWhitty/remote-ppg',
      license='MIT',
      setup_requires=['pytest-runner',],
      tests_require=['pytest', 'python-coveralls', 'coverage'],
      install_requires=[
          "numpy",
          "scipy"
      ],
      packages = ['ppg_features'],
      include_package_data=True,
      scripts=['ppg_features/ppg_features.py'],
              
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Other Audience',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
)
