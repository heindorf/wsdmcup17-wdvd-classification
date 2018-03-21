# -----------------------------------------------------------------------------
# WSDM Cup 2017 Classification and Evaluation
#
# Copyright (c) 2017 Stefan Heindorf, Martin Potthast, Gregor Engels, Benno Stein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------


class Feature:
    def __init__(self, input_names, transformers=None, output_name=None):
        if transformers is None:
            transformers = []

        if type(input_names) is not list:
            input_names = [input_names]
        if type(transformers) is not list:
            raise Exception("transformers should be a list, but it was " +
                            str(type(transformers)))

        self.__input_names = input_names
        self.__transformers = transformers

        self.__output_name = output_name
        if self.__output_name is None:
            if len(input_names) > 1:
                raise Exception("There should only be one input name, " +
                                "if no output name is specified.")
            self.__output_name = str(input_names[0])

        self.__group = None
        self.__subgroup = None

    def get_input_names(self):
        return self.__input_names

    def get_output_name(self):
        return self.__output_name

    def get_transformers(self):
        return self.__transformers

    def get_group(self):
        return self.__group

    def get_subgroup(self):
        return self.__subgroup

    def set_group(self, group):
        self.__group = group

    def set_subgroup(self, subgroup):
        self.__subgroup = subgroup

    def __str__(self):
        return ("Feature %s (%s, %s)" %
                (self.__output_name,
                 str(self.__input_names),
                 str(self.__transformers)))
