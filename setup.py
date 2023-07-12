from setuptools import setup,find_packages

def get_requirements(filepath:str):
    requirements = []
    with open(filepath,'r') as f:
        requirements = f.readlines()
    
    hyphen_e_dot = '-e .\n'
    clearList = []
    
    for i in range(len(requirements)):
        word = requirements[i]
        if word == hyphen_e_dot:
            pass
        else: 
            word = word.replace('\n',"")
            clearList.append(word)
    return clearList

setup(
    name= 'DiamondPricePredicion-package',
    version= '0.0.1',
    author_email= 'mayur@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    package = find_packages(),
)