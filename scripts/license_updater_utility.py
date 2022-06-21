# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
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
from argparse import ArgumentParser
from pathlib import Path
import os


target_dirs = ['deap_er', 'scripts', 'tests']
license_file = 'LICENSE'
default_frame_len = 90
default_min_padding = 3


# -------------------------------------------------------------------------------------- #
def read_license(path):
    lic_file = path.with_name(license_file)
    if not lic_file.exists():
        raise FileExistsError('License file not found.')
    with open(lic_file, 'r', encoding='utf-8') as f:
        _lic = f.read()
    return _lic


# -------------------------------------------------------------------------------------- #
def frame_license(_lic: str) -> str:
    def frame(_line):
        if len(_line) > args.frame_len - (args.min_padding * 2 + 2):
            raise OverflowError('License line is longer than the allowed frame length.')
        right_padding = (args.frame_len - (1 + args.min_padding + len(_line))) - 1
        _newline = '#' + (' ' * args.min_padding) + _line + (' ' * right_padding) + '#\n'
        return _newline

    _lic = _lic.split('\n')
    frame_line = '# ' + ('=' * (args.frame_len - 4)) + ' #'
    framed_lic = ''.join(frame_line + '\n')
    framed_lic += frame('')
    for line in _lic:
        framed_lic += frame(line)
    framed_lic += frame_line
    return framed_lic


# -------------------------------------------------------------------------------------- #
def update_licenses(parent: Path) -> None:
    for child in parent.iterdir():
        child = child.resolve()
        if child.exists():
            if child.is_dir():
                update_licenses(child)
            elif child.is_file() and child.suffix == ".py":
                with open(child, 'r', encoding='utf-8') as f:
                    file = f.read()
                    file = file.split('\n')
                    for line in tuple(file):
                        if line.startswith('#'):
                            file.pop(0)
                        else:
                            break
                    file = lic.split('\n') + file
                    file = '\n'.join(file)
                with open(child, 'w', encoding='utf-8') as f:
                    f.write(file)
                print(f'Updated license in: \'{child}\'')


# -------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--frame-len',
        default=default_frame_len,
        required=False,
        type=int
    )
    parser.add_argument(
        '--min-padding',
        default=default_min_padding,
        required=False,
        type=int
    )
    args = parser.parse_known_args()[0]

    cwd = Path(os.getcwd())
    lic = read_license(cwd)
    lic = frame_license(lic)

    for tgt in target_dirs:
        work_dir = cwd.with_name(tgt)
        update_licenses(work_dir)
