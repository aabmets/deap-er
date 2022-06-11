# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 The Original DEAP Team, Mattias Aabmets and Contributors          #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


# Compilation of C module in c_lib
# com_args = ['-std=c99', '-O3', '-fopenmp']
# link_args = ['-fopenmp']
# extra_compile_args={'gcc': com_args},
              # extra_compile_args=com_args,
              # extra_link_args=link_args,
              # include_dirs=[numpy.get_include()]

hv_module = Extension(
    "deap_er.hypervolume", sources=[
        "deap_er/hypervolume/c_ext/_hv.c",
        "deap_er/hypervolume/c_ext/_hv.cpp"
    ]
)
ext_modules = [hv_module]


class BuildExt(build_ext):

    def run(self):
        try:
            print('running')
            build_ext.run(self)
        except Exception as e:
            print(e)

    def build_extensions(self):
        try:
            print('building extensions')
            super().build_extensions()
        except Exception as e:
            print(e)


def build(setup_kwargs):
    setup_kwargs.update(
        dict(
            cmdclass=dict(build_ext=BuildExt),
            ext_modules=ext_modules,
            zip_safe=False
        )
    )

print('inside build file')