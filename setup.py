from setuptools import setup, find_packages

setup( name = "stylometry",
	version = "1.0",
	author = 'Chris Ward',
	author_email = 'chrisward@email.com',
	packages=find_packages(include=['stylometry','stylometry.*']),
	install_requires = ['numpy>=1.26.0','pandas>=2.1.1'])