import setuptools
from typing import List
#from setuptools import find_packages,setup  # Import necessary functions from setuptools for package setup.


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "kidney_disease_cassification"
AUTHOR_USER_NAME = "ayushkhanal1"
SRC_REPO = "Classifier"
AUTHOR_EMAIL = "ayukhanalsh100@gmail.com"

hyphen_dot_e="-e ."  
def get_requirements(file_path:str) ->List[str]:  # Define a function to read requirements from a file.
    requirements=[]  # Initialize an empty list for requirements.
    with open(file_path) as temp_file:  # Open the file in read mode.
        requirements=temp_file.readlines()  # Read all lines into the list.
        requirements=[req.replace("\n","") for req in requirements]  # Remove newline characters from each line.

        if hyphen_dot_e in requirements:  # Check if the editable flag is in the list.
            requirements.remove(hyphen_dot_e)  # Remove it if present.

    return requirements  # Return the cleaned list of requirements.



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=get_requirements('requirements.txt')
    )

