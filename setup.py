from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Modeling the 3d genome',
    'download_url': 'https://github.com/kundajelab/genome3d',
    'version': '0.1.0',
    'packages': ['genome3d'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==0.3.2', 'sklearn', 'future',
                         'psutil', 'joblib'],
    'dependency_links': [],
    'scripts': [],
    'name': 'genome3d'
}

if __name__== '__main__':
    setup(**config)
