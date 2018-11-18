from setuptools import setup
import Backend
import Frontend


setup (
    name='Frontend + Backend',
    packages=['Frontend', 'Backend'],
    include_package_data=True,
    install_requires=[
        'flask',
    ]
)