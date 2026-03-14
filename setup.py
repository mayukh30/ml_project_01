# responsible in creating my ml application as a package and also responsible for installing the dependencies in my project

from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->list[str]:
    # this function will return list of requirements which are present in the requirements.txt file
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="ml_project",
    version="0.0.1",
    author="Mayukh",
    author_email="mayukhs.it.ug@jadavpuruniversity.in",
    packages=find_packages(),
# find_packages() will automatically find all the packages in my project and include them in the distribution
# it check how many __init__.py files are there in my project and it will include all the packages in my project

    install_requires=get_requirements('requirements.txt')


)