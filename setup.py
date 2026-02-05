# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from glob import glob

# Define package metadata
package_name = 'pycontree'
extension_name = 'ccontree'
__version__ = "1.0.6"

ext_modules = [
    Pybind11Extension(package_name + '.' + extension_name,
        sorted(glob("code/*/src/*.cpp", recursive = True) + ["code/binding.cpp"]),
        include_dirs = ["code/DataStructures/include", "code/Engine/include", "code/Utilities/include"],
        define_macros = [('VERSION_INFO', __version__)], # passing in the version to the compiled code
        language='c++',
        cxx_std=17
    )
]

setup(
    name=package_name,
    version=__version__,
    ext_modules=ext_modules,
    dev_requires=[],
    install_requires=['pandas', 'numpy'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)