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
from typing import Union
from pathlib import Path
import subprocess
import os


# -------------------------------------------------------------------------------------- #
"""
Running this module from the terminal will convert all the old DEAP 
Python 2.7 files to Python 3 syntax. Total conversion is not guaranteed.
Terminal CWD must be the main project folder when running this module.
"""


# -------------------------------------------------------------------------------------- #
def remove_bak_file(path: Path) -> None:
    bak_file = path.with_suffix(".py.bak")
    if bak_file.exists():
        os.remove(bak_file)


# -------------------------------------------------------------------------------------- #
def convert_files(parent: Path, script: Path) -> None:
    for child in parent.iterdir():
        child = child.resolve()
        if child.exists():
            if child.is_dir():
                convert_files(child, script)
            elif child.is_file() and child.suffix == ".py":
                cmd = ["python", str(script), "-w", str(child)]
                subprocess.run(cmd)
                remove_bak_file(child)


# -------------------------------------------------------------------------------------- #
def find_2to3_script() -> Union[Path, None]:
    env_paths = os.getenv("PATH")
    env_paths = env_paths.split(";")
    for ep in env_paths:
        if "python" in ep.lower():
            script_path = Path(ep).joinpath("Tools/scripts/2to3.py")
            if script_path.resolve().exists():
                return script_path


# -------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    script_file = find_2to3_script()
    working_dir = Path(os.getcwd()).joinpath("__old-deap__").resolve()
    convert_files(working_dir, script_file)
