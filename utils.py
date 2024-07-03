import collections
from collections import OrderedDict
import re
import argparse
import os.path
import sys
from pathlib import Path
from sksequitur import parse, Mark, Parser, Grammar
import pickle
import numpy as np
from matplotlib import pyplot as plt


class Parameter():
    def __init__(self, paramNumber, value, _type):
        self.paramNumber = paramNumber
        self.value = value
        self._type = _type

    def __repr__(self):
        a = "P{" + (str)(self.paramNumber) + "}"
        return a

    def __str__(self):
        a = "P{" + (str)(self.paramNumber) + "}"
        return a


def find_nearest(array, value, k):
    array = np.asarray(array)
    idx = np.argsort(abs(array - value))[:k]
    return array[idx]


def parameterize(mystring):
    from collections import OrderedDict
    angleCommands = set(['-', '+', '^', '&', '\\', '/'])
    cmdIdx = None
    opIdx = None
    clIdx = None
    isDist = False
    isAngle = False
    dist = OrderedDict()
    angle = OrderedDict()
    params = OrderedDict()

    afterReplace = ""

    for i in range(len(mystring)):
        letter = mystring[i]
        if(letter == 'F'):
            cmdIdx = i
            isDist = True
        if(letter in angleCommands):
            cmdIdx = i
            isAngle = True
        if(letter == '('):
            opIdx = i
        if(letter == ')'):
            clIdx = i
            sub = mystring[opIdx+1:clIdx]
            if(isDist):
                if(sub not in dist):
                    dist[sub] = len(params)
                    param = Parameter(len(params), float(sub), "D")
                    params[len(params)] = param
                paramNumber = dist[sub]
                afterReplace += f"{mystring[cmdIdx]}({params[paramNumber]})"
                isDist = False
            if(isAngle):
                if(sub not in angle):
                    angle[sub] = len(params)
                    param = Parameter(len(params), float(sub), "A")
                    params[len(params)] = param
                paramNumber = angle[sub]
                afterReplace += f"{mystring[cmdIdx]}({params[paramNumber]})"
                isAngle = False
        if(letter == '[' or letter == ']'):
            afterReplace += letter
    return params, afterReplace


def addRule(nextRule, ruleCnt, rules, mystring=None):
    if(nextRule == None):
        exp = mystring
        rule = Rule(ruleCnt, exp)
        rules[0] = exp
    else:
        rule = Rule(ruleCnt, nextRule)
        rules[ruleCnt] = nextRule
        ruleCnt += 1
    return ruleCnt, rule


def flatten(xs):
    res = []

    def loop(ys):
        for i in ys:
            if isinstance(i, list):
                loop(i)
            else:
                res.append(i)
    loop(xs)
    return res


def get_rules_grammar(grammar):
    rules = dict()
    for i in range(len(grammar)):
        elems = []
        for idx, elem in enumerate(grammar[i]):
            try:
                elems.append((int)(elem))
            except ValueError:
                elems.append(f"{elem}")
        rules[i] = elems
    return rules


def isConverted(rule):
    intCount = 0
    for val in rule:
        if(isinstance(val, int) or isinstance(val, list)):
            intCount += 1
    return intCount == 0


def convert(rule, rules):
    for idx, val in enumerate(rule):
        if(isinstance(val, int)):
            rule[idx] = rules[val]
    return flatten(rule)


def reverseParameter(seq, params):
    for idx in params:
        param = f"P{{{idx}}}"
        seq = seq.replace(param, f"{params[idx].value:.3f}")
    return seq

def reverseParameterClustering(seq, params):
    for idx in params:
        param = f"P{{{idx}}}"
        seq = seq.replace(param, f"{params[idx].value}")
    return seq


def reFormatString(sequence):
    cmd = ""
    arr = []
    for letter in sequence:
        cmd += letter
        if letter == ')' or letter == '[' or letter == ']':
            arr.append(cmd)
            cmd = ""
    return arr


def save_grammar_rule(path, parameterized, reConstructed, rulesPrint, paramsPrint, params):
    path = Path(path)
    parent = path.parent
    fileName = path.name
    extension = path.suffix

    newFileName = "v5_" + fileName.split(extension)[0]+"_grammar.txt"
    newPath = parent.joinpath(newFileName)
    print(f"Grammar is saved at {str(newPath)}")
    with open(str(newPath), 'w') as the_file:
        for rule in rulesPrint:
            the_file.write(f"{rule}\n")

    newFileParameter = "v5_" + fileName.split(extension)[0]+"_parameters.txt"
    newPathParameter = parent.joinpath(newFileParameter)
    print(f"Parameters are saved at {str(newPathParameter)}")
    with open(str(newPathParameter), 'w') as the_file:
        for parameter in paramsPrint:
            the_file.write(f"{parameter}\n")

    newFileParameterPickle = "v5_" + \
        fileName.split(extension)[0]+"_parameters_obj.pickle"
    newPathParameterPickle = parent.joinpath(newFileParameterPickle)
    print(f"Parameter Pickle is saved at {str(newPathParameterPickle)}")
    with open(str(newPathParameterPickle), 'wb') as the_file:
        pickle.dump(params, the_file)

    newFileNameGenSeq = "v5_" + \
        fileName.split(extension)[0]+"_reconstructed_string.lstring"
    newPathGenSeq = parent.joinpath(newFileNameGenSeq)
    print(f"L-system string is re-constructed at {str(newPathGenSeq)}")
    with open(str(newPathGenSeq), 'w') as the_file:
        for command in reConstructed:
            the_file.write(f"{command}\n")

    newFileNameParameterized = "v5_" + \
        fileName.split(extension)[0]+"_parameterized.txt"
    newPathParameterized = parent.joinpath(newFileNameParameterized)
    print(f"L-system string is parameterized at {str(newPathParameterized)}")
    with open(str(newPathParameterized), 'w') as the_file:
        the_file.write(f"{parameterized}\n")


def get_grmmar(grammar):
    rules = dict()
    seq = []
    for i in range(len(grammar)):
        line = ""
        line += f"{i}"
        line += "->"
        for idx, elem in enumerate(grammar[i]):
            if idx == len(grammar[i]) - 1:
                line += f"{elem}"
            else:
                line += f"{elem} "
        seq.append(line)
    return seq


def getBins(a, numOfBin=10):
    offSet = (a.max()-a.min())/numOfBin
    bins = []
    binValue = a.min()
    while binValue < a.max():
        bins.append(binValue)
        binValue += offSet
    return bins


def get_statistics(grammar, params):
    # - Get # of rules
    # - Get # of symbols
    # - Get # of iterations (Ex) rule1 can be appeared multiple times.)
    import sksequitur
    numOfRules = len(grammar)  # Get # of rules
    commands = set(['F', '-', '+', '^', '&', '\\', '/', '[', ']'])
    numOfSymbols = 0  # - Get # of symbols
    for i in range(len(grammar)):
        g = grammar[i]
        for unit in g:
            if unit in commands:
                numOfSymbols += 1
    numIteration = 0  # - Get # of iterations
    for i in range(len(grammar)):
        g = grammar[i]
        for unit in g:
            if type(unit) == sksequitur.api.Production:
                numIteration += 1
    statistics = dict()
    statistics['rules'] = numOfRules
    statistics['symbols'] = numOfSymbols
    statistics['iterations'] = numIteration
    return statistics


def get_statistics(grammar, params):
    # - Get # of rules
    # - Get # of symbols
    # - Get # of iterations (Ex) rule1 can be appeared multiple times.)
    import sksequitur
    numOfRules = len(grammar)  # Get # of rules
    commands = set(['F', '-', '+', '^', '&', '\\', '/', '[', ']'])
    numOfSymbols = 0  # - Get # of symbols
    for i in range(len(grammar)):
        g = grammar[i]
        for unit in g:
            if unit in commands:
                numOfSymbols += 1
    numIteration = 0  # - Get # of iterations
    for i in range(len(grammar)):
        g = grammar[i]
        for unit in g:
            if type(unit) == sksequitur.api.Production:
                numIteration += 1
    statistics = dict()
    statistics['rules'] = numOfRules
    statistics['symbols'] = numOfSymbols
    statistics['iterations'] = numIteration
    return statistics
