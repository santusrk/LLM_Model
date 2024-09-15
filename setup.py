from setuptools import find_packages ,setup
from typing import List

def get_requirements(filepath:str) -> List[str]:
    '''
    this function requires list of requirements
    '''
    requirements=[]
    HYPEN = '-e .'
    with open(filepath) as file:
        requirements=file.readlines()
        requirements= [req.replace("\n","") for req in requirements]

        if HYPEN in requirements:
            requirements.remove(HYPEN)

    return requirements



setup(
    name = "LLM Project",
    version = '0.0.1',
    author = "Srikant Nayak",
    author_email= "srikantnayak.2010@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)