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
from collections import defaultdict
from itertools import chain


__all__ = ['Logbook']


# ====================================================================================== #
class Logbook(list):
    def __init__(self):
        self.chapters = defaultdict(Logbook)
        self.buff_index: int = 0
        self.log_header: bool = True
        self.columns_len: list = list()
        self.header: list = list()
        super().__init__()

    # -------------------------------------------------------------------------------------- #
    @property
    def stream(self):
        start_index, self.buff_index = self.buff_index, len(self)
        return self.__str__(start_index)

    # -------------------------------------------------------------------------------------- #
    def record(self, **infos):
        apply_to_all = {k: v for k, v in infos.items() if not isinstance(v, dict)}
        for key, value in infos.items():
            if isinstance(value, dict):
                chapter_infos = value.copy()
                chapter_infos.update(apply_to_all)
                self.chapters[key].record(**chapter_infos)
                del infos[key]
        self.append(infos)

    # -------------------------------------------------------------------------------------- #
    def select(self, *names):
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return tuple([entry.get(name, None) for entry in self] for name in names)

    # -------------------------------------------------------------------------------------- #
    def pop(self, index=0):
        if index < self.buff_index:
            self.buff_index -= 1
        return super(self.__class__, self).pop(index)

    # -------------------------------------------------------------------------------------- #
    def __delitem__(self, key):
        if isinstance(key, slice):
            for i, in range(*key.indices(len(self))):
                self.pop(i)
                for chapter in self.chapters.values():
                    chapter.pop(i)
        else:
            self.pop(key)
            for chapter in self.chapters.values():
                chapter.pop(key)

    # -------------------------------------------------------------------------------------- #
    def __txt__(self, start_index: int) -> list:
        columns = self.header
        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len: list = list(map(len, columns))

        chapters_txt = {}
        offsets = defaultdict(int)
        for name, chapter in self.chapters.items():
            chapters_txt[name] = chapter.__txt__(start_index)
            if start_index == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)

        str_matrix = []
        for i, line in enumerate(self[start_index:]):
            str_line = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i+offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:n}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                self.columns_len[j] = max(self.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)

        if start_index == 0 and self.log_header:
            n_lines = 1
            if len(self.chapters) > 0:
                n_lines += max(map(len, chapters_txt.values())) - len(self) + 1
            header = [[] for _ in range(n_lines)]
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    length = max(len(line.expandtabs()) for line in chapters_txt[name])
                    blanks = n_lines - 2 - offsets[name]
                    for i in range(blanks):
                        header[i].append(" " * length)
                    header[blanks].append(name.center(length))
                    header[blanks+1].append("-" * length)
                    for i in range(offsets[name]):
                        header[blanks+2+i].append(chapters_txt[name][i])
                else:
                    length = max(len(line[j].expandtabs()) for line in str_matrix)
                    for line in header[:-1]:
                        line.append(" " * length)
                    header[-1].append(name)
            str_matrix = chain(header, str_matrix)

        template = "\t".join("{%i:<%i}" % (i, l) for i, l in enumerate(self.columns_len))
        str_list = [template.format(*line) for line in str_matrix]
        return str_list

    # -------------------------------------------------------------------------------------- #
    def __str__(self, start_index: int = 0):
        text = self.__txt__(start_index)
        return "\n".join(text)
