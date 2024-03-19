import multiprocessing as mp
from multiprocessing import Pool
import os
import json
from glob import glob
import math
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import re
import json
from collections import defaultdict
import gzip
from collections import Counter
import gc
import time
import string
from tqdm import tqdm
import math
from utils import *


clustering_ratio = 0.1
tree = ""

def process_file_cluster(path):
    clusteredFolderPath = f"{tree}Clustered"
    picklePath = f"{tree}Pickle"
    os.makedirs(clusteredFolderPath, exist_ok=True)
    uniqueValues = set()
    clusterNumTotal = 0
    with open(path, "r") as f:
        sequence = "".join(line.strip() for line in f)
    path = os.path.basename(path)
    fileName = path.split(os.sep)[-1]
    params, mystring = parameterize(sequence)
    commands = set(["F", "-", "+", "^", "&", "\\", "/", "[", "]"])
    parser = Parser()
    nonCmd = ""
    for c in mystring:
        if c in commands:
            data = [nonCmd]
            parser.feed(data)
            nonCmd = ""
            parser.feed(c)
        else:
            nonCmd += c
    grammar = Grammar(parser.tree)
    rulesFromGrammar = get_rules_grammar(grammar)

    distValue = []
    angleValue = []
    values = []

    for p in params:
        param = params[p]
        val = round(param.value, 2)
        if param._type == "D":
            distValue.append(val)
            values.append(val)
        elif param._type == "A":
            angleValue.append(val)
            values.append(val)

    paramTable = dict()  # param.value : id
    paramTableDist = dict()
    table = dict()
    for idx, p in enumerate(params):
        if params[p]._type == "A":
            key = float(params[p].value)
            key = round(key, 2)
            try:
                paramTable[key].add(idx)
                table[key].add(idx)
            except:
                paramTable[key] = set()
                paramTable[key].add(idx)
                table[key] = set()
                table[key].add(idx)

        if params[p]._type == "D":
            key = float(params[p].value)
            key = round(key, 2)
            try:
                paramTable[key].add(idx)
                table[key].add(idx)
            except:
                paramTable[key] = set()
                paramTable[key].add(idx)
                table[key] = set()
                table[key].add(idx)

    UNIQUE_ANGLES = len(set(angleValue))
    UNIQUE_DIST = len(set(distValue))
    NUM = int(len(set(values)) * clustering_ratio)
    clusterNumTotal += NUM
    cluster = KMeans(n_clusters=NUM)
    angleValue = np.asarray(angleValue)
    np_values = np.asarray(values)
    ids = cluster.fit_predict(np_values.reshape(-1, 1))
    KmeanValues = cluster.cluster_centers_
    clusterLabels = cluster.labels_
    b4, after = [], []
    for i in range(0, len(np_values)):
        angle = np_values[i]
        # get params id using angle
        ids = table[angle]
        # Get cluster id
        cluster_id = clusterLabels[i]
        # Get KmeanValue using cluster id
        clusterVal = round(KmeanValues[cluster_id][0], 2)
        uniqueValues.add(clusterVal)
        # Update the param value using the param id
        for id in ids:
            b4.append(params[id].value)
            params[id].value = clusterVal
            after.append(math.degrees(params[id].value))

    rulesFromGrammar = get_rules_grammar(grammar)
    firstRule = rulesFromGrammar[0]
    while not isConverted(firstRule):
        firstRule = convert(firstRule, rulesFromGrammar)
    parameterized = "".join(firstRule)

    reConstructed = reverseParameterClustering(parameterized, params)
    reConstructedFormatted = reFormatString(reConstructed)

    fileName = path.split(".")[0]
    newFileNameGenSeq = f"./{clusteredFolderPath}/{fileName}.lstring"
    with open(str(newFileNameGenSeq), "w") as the_file:
        for command in reConstructedFormatted:
            the_file.write(f"{command}\n")
    import pickle

    pickle_path = os.path.join(picklePath, fileName + ".pkl")
    os.makedirs(picklePath, exist_ok=True)
    with open(pickle_path, "wb") as fp:
        pickle.dump(uniqueValues, fp)
    return None



def number2letter(n, b=string.ascii_uppercase):
    d, m = divmod(n, len(b))
    return number2letter(d - 1, b) + b[m] if d else b[m]


def letter2number(number):
    col_number = (
        sum([(ord(number.upper()[-i - 1]) - 64) * 26 ** i for i in range(len(number))])
        - 1
    )
    return col_number


def get_table(path):
    with open(path, "rb") as f:
        table = pickle.load(f)
    # clusterVal to alphabet ID
    alphabetTable = dict()
    for idx, val in enumerate(sorted(table)):
        alphabetTable[val] = number2letter(idx)
    return alphabetTable


def convertLetter(path, combined_pickle_path):
    convertedPath = f"{tree}Converted"
    os.makedirs(convertedPath, exist_ok=True)
    with open(path, "r") as f:
        sequence = "".join(line.strip() for line in f)
    path = os.path.basename(path)
    fileName = path.split(os.sep)[-1]
    params, mystring = parameterize(sequence)
    commands = set(["F", "-", "+", "^", "&", "\\", "/", "[", "]"])

    ## Pickle combined?
    alphabetTable = get_table(combined_pickle_path)
    # replace values
    for p in params:
        value = params[p].value
        letterValue = alphabetTable[value]
        params[p].value = letterValue
    parser = Parser()
    nonCmd = ""
    for c in mystring:
        if c in commands:
            data = [nonCmd]
            parser.feed(data)
            nonCmd = ""
            parser.feed(c)
        else:
            nonCmd += c
    grammar = Grammar(parser.tree)
    rulesFromGrammar = get_rules_grammar(grammar)
    firstRule = rulesFromGrammar[0]
    while not isConverted(firstRule):
        firstRule = convert(firstRule, rulesFromGrammar)
    parameterized = "".join(firstRule)

    reConstructed = reverseParameterClustering(parameterized, params)
    reConstructedFormatted = reFormatString(reConstructed)

    fileName = path.split(".")[0]
    newFileNameGenSeq = f"{convertedPath}/{fileName}.lstring"
    with open(str(newFileNameGenSeq), "w") as the_file:
        for command in reConstructedFormatted:
            the_file.write(f"{command}\n")


def combine_all_pickles(tree, picklePath):
    pickles = glob(f"./{picklePath}/*")
    table = set()
    for p in pickles:
        p_val = pickle.load(open(p, "rb"))
        table = table | p_val

    ratio_int = int(clustering_ratio * 100)
    combined_pickle_name = f"{tree}_{len(pickles)}_set_ratio{ratio_int}.pkl"
    with open(combined_pickle_name, "wb") as f:
        pickle.dump(table, f)
    return combined_pickle_name




def getPair(ls):
    _op, _cl = [], []
    brakcetList = []

    for idx, l in enumerate(ls):
        if l == "[":
            _op.append(idx)
            brakcetList.append(("[", idx))
        if l == "]":
            _cl.append(idx)
            brakcetList.append(("]", idx))
    stck = []
    pair = dict()
    for ele1, ele2 in brakcetList:
        if "[" in ele1:
            stck.append((ele1, ele2))
        elif "]" in ele1:
            # res.append((stck.pop()[1], ele2))
            # s,e = res[-1]
            s, e = stck.pop()[1], ele2
            pair[s] = e
    return pair


def isBracketFree(lstring):
    if "[" in lstring and "]" in lstring:
        return True
    else:
        return False


def getGraveliusOrder(string, index):
    order = 0
    for s in string[: index + 1]:
        if s == "[":
            order += 1
        elif s == "]":
            order -= 1
    return order


def readLStrings(path):
    with open(path) as file:
        lines = file.readlines()
    strings = ""
    for l in lines:
        l = l.rstrip()
        strings += l
    return strings


def getMaxGraveliusOrder(string):
    order = 0
    maxOrder = -1
    for s in string:
        if s == "[":
            order += 1
            if maxOrder < order:
                maxOrder = order
        elif s == "]":
            order -= 1
    return maxOrder


def getRuleName(order, newRules):
    name = f"R({order}@{len(newRules)})"
    return name


def get_values(path):
    orderPath = f"{tree}Orders"
    tablePath = f"{tree}Tables"

    os.makedirs(orderPath, exist_ok=True)
    os.makedirs(tablePath, exist_ok=True)

    ruleValue = dict()  # value : ruleNumber -> Table for the rule value check
    ruleTable = dict()  # ruleNumber : value
    orderValue = defaultdict(list)  # order : [actual values]
    # lstring = '+F[F[+F]F[-F]F[-F[+F]F]]'
    lstring = readLStrings(path)
    deepestOrder = getMaxGraveliusOrder(lstring)
    deepestOrder += 1
    newRules = []  # Save the new rules in this iteration
    newRuleOrder = dict()
    for order in range(deepestOrder):
        newRuleOrder[str(order)] = []
    while isBracketFree(lstring):
        pairs = getPair(lstring)  # start : end
        # newRules = []  # Save the new rules in this iteration
        deepOrder = -1
        orderTable = defaultdict(list)  # order : [start indicies]
        for start in pairs:
            order = getGraveliusOrder(lstring, start)
            if order > deepOrder:
                deepOrder = order
            orderTable[order].append(start)

        # Go to the deepest order
        starts = orderTable[deepOrder]
        newOrder = deepestOrder - deepOrder

        for start in starts:
            end = pairs[start]
            value = lstring[start : end + 1]
            ruleNumber = -1
            key = len(ruleTable)
            rule = f"R({key})"
            rule = getRuleName(newOrder, newRules)
            rule = getRuleName(newOrder, newRuleOrder[str(newOrder)])
            try:
                # Already in the table
                ruleNumber = ruleValue[value]
                ruleTable[rule] = value
            except:
                # New value
                ruleTable[rule] = value
                ruleValue[value] = rule
                newRuleOrder[str(newOrder)].append(value)
                orderValue[str(newOrder)].append(value)
            # Get the values from the gavelius order

        for newRule in newRuleOrder[str(newOrder)]:
            lstring = lstring.replace(newRule, ruleValue[newRule])

    # Take the left one
    key = len(ruleTable)
    rule = f"R({key})"
    rule = getRuleName(newOrder, newRuleOrder[str(newOrder)])

    last = f"[{lstring}]"
    ruleTable[rule] = last
    ruleValue[last] = rule
    newRuleOrder[str(newOrder)].append(lstring)
    orderValue[deepestOrder - (len(orderValue) + 1)].append(last)

    orderTable = dict()
    for key in orderValue.keys():
        values = orderValue[key]
        table = defaultdict(list)
        for idx, val in enumerate(values):
            table[idx].append(val)
        orderTable[key] = table

    name = os.path.basename(path)
    name = name.split(".")[0]
    with open(f"./{tablePath}/{tree}Table_{name}.json", "w") as outfile:
        json.dump(orderTable, outfile, sort_keys=False, indent=4)
    with open(f"./{orderPath}/{tree}Order_{name}.json", "w") as outfile:
        json.dump(orderValue, outfile, sort_keys=False, indent=4)
    return 1


def get_rules(strings):
    rules = re.findall(r"R\(\w+@\w+\)", strings)
    return rules


def get_Rxy(rule):
    x = rule[rule.index("(") + 1 : rule.index("@")]
    y = rule[rule.index("@") + 1 : rule.index(")")]
    return x, y


# def get GeoInfo


def get_geoInfo(strings):
    geoInfo = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for s in strings:
        if s in symbols:
            geoInfo += s
    return geoInfo


def get_geoInfoCount(strings):
    count = {"^": 0, "F": 0, "R": 0, "+": 0, "/": 0, "&": 0, "\\": 0, "-": 0}

    for key in count.keys():
        count[key] = strings.count(key)
    keys = list(count.keys())
    keys.sort()
    res = ""
    for key in keys:
        res += f"{key}{count[key]} "
    # print(res)
    res = res.rstrip()
    return res


def getSubRuleToken(strings):
    token = ""
    for st in strings:
        token += f"R({st})"
    return token


def getToken(tree, key, idx, rule, geoInfo):
    if tree == "Maple":
        treeToken = "M"
    if tree == "Acacia":
        treeToken = "A"
    if tree == "Birch":
        treeToken = "B"
    if tree == "Oak":
        treeToken = "O"
    if tree == "Pine":
        treeToken = "P"

    start = f"<{treeToken}{key}@{idx} {geoInfo}>"
    if geoInfo == "":
        start = f"<{treeToken}{key}@{idx}>"

    newSeed = f"{start}"
    newSeed += f"{rule}"
    newSeed += "<END>"
    return newSeed


def get_value(values):
    temp = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for idx, s in enumerate(values):
        temp += s
        if s == ")":
            temp += " "
    temp = temp.rstrip()
    return temp


def get_train_data(files):
    trainDataPath = f"{tree}TrainData"
    if tree == "Maple":
        treeToken = "M"
    if tree == "Acacia":
        treeToken = "A"
    if tree == "Birch":
        treeToken = "B"
    if tree == "Oak":
        treeToken = "O"
    if tree == "Pine":
        treeToken = "P"
    os.makedirs(trainDataPath, exist_ok=True)
    for _file in tqdm(files):
        basename = os.path.basename(_file)
        trainData = defaultdict(list)
        with open(_file) as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
        keys.sort(key=int)
        valueLinkTable = dict()
        valueTable = dict()

        orderTable = data
        originTable = dict()  # RuleNumber : Its origirn Rule Number
        geoInfoTable = dict()  # RuleNumber: Its geo information
        orderTable.keys()
        for order in orderTable.keys():
            orders = orderTable[order]
            for key in orders.keys():
                rules = get_rules(orders[key][0])
                geoInfoTable[f"R({order}@{key})"] = orders[key][0]
                for r in rules:
                    x, y = get_Rxy(r)
                    originTable[f"R({x}@{y})"] = f"R({order}@{key})"
        for order in keys:
            orderValues = data[order]
            for idx in orderValues.keys():
                vals = orderValues[idx][0]
                valueTable[f"{order}@{idx}"] = vals
                spaceGeoInfo = ""
                currentRule = f"R({order}@{idx})"
                if currentRule != "R(0@0)":
                    origin = originTable[f"R({order}@{idx})"]
                    geoInfo = geoInfoTable[origin]
                    spaceGeoInfo = " ".join(list(geoInfo))
                    spaceGeoInfo = get_value(geoInfo[1:-1])
                token = getToken(tree, order, idx, vals, spaceGeoInfo)
                trainData[order].append(token)
        for order in trainData.keys():
            temp = []
            for value in trainData[order]:
                temp_v = value.replace("R(", f"{treeToken}(")
                temp.append(temp_v)
            trainData[order] = temp
        with open(f"./{trainDataPath}/{basename}", "w") as outfile:
            json.dump(trainData, outfile, sort_keys=True, indent=4)



def preprocess_end_tokens(end_info):
    tokens = []
    temp = ""
    for c in end_info:
        if c in ["[", "]"]:
            tokens.append(c)
            continue
        temp += c
        if c == ")":
            tokens.append(temp)
            temp = ""
    tokens.append("[END]")
    return " ".join(tokens)


def get_start_data_tokens(data):
    open_bracket_index = data.index("[")
    start_tokens = data[:open_bracket_index]
    end_tokens = preprocess_end_tokens(data[open_bracket_index:])
    return start_tokens, end_tokens

def split_to_train_test(files):
    csvFilePath = f"{tree}CSV"
    os.makedirs(csvFilePath, exist_ok=True)
    t_start_list, t_end_list = [], []
    for f in tqdm(files[:]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if(order == "0"):
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        df = pd.DataFrame(list(zip(starts_list, ends_list)), columns=["SRC", "TRG"])
        os.makedirs(f"{csvFilePath}", exist_ok=True)
        df.to_csv(f"./{csvFilePath}/{basename}.csv", index=False)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)

    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_total.csv", index=False)

    starts_counter = []
    s_counter = Counter()
    for start_list in tqdm(t_start_list):
        s = start_list.split()
        s_counter.update(s)
    del starts_counter
    gc.collect()

    ends_counter = []
    e_counter = Counter()
    for end_list in tqdm(t_end_list):
        e = end_list.split()
        e_counter.update(e)
    del ends_counter
    gc.collect()

    with gzip.open(f"./{tree}_S_counter_2.pkl", "wb") as f:
        pickle.dump(s_counter, f)

    with gzip.open(f"./{tree}_E_counter_2.pkl", "wb") as f:
        pickle.dump(e_counter, f)


    t_start_list, t_end_list = [], []
    for f in tqdm(files[:-100]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if(order == "0"):
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)
    ### HERE

    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_train_total.csv", index=False)

    train_len = int(len(files) * 0.8)
    t_start_list, t_end_list = [], []
    for f in tqdm(files[-100:]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if(order == "0"):
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)

    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_test_total.csv", index=False)


def start_process(t):
    print(f"Preprocessing starts with {t}")
    global tree
    tree = t
    start_time = time.time()
    picklePath = f"{tree}Pickle"
    clusteredFolderPath = f"{tree}Clustered"
    convertedPath = f"{tree}Converted"
    orderPath = f"{tree}Orders"
    tablePath = f"{tree}Tables"
    csvFilePath = f"{tree}CSV"
    trainDataPath = f"{tree}TrainData"
    print(f"[1] Processing Clustering, path: {tree}")
    files = glob(os.path.join(tree, "*.lstring"))[:300  ]
    sorted_files = sorted(files)
    p = mp.Pool(mp.cpu_count())
    for path in sorted_files:
        p.apply_async(process_file_cluster, args=(path,))
    p.close()
    p.join()
    
    with Pool(mp.cpu_count()) as p:
        p.map(process_file_cluster, files)

    print(f"[1] Completed!, Processing Clustering, path: {t}")

    print("[2] Now, combining all pickle files")
    combined_pickle_path = combine_all_pickles(tree, picklePath)
    clusteredFiles = glob(f"{clusteredFolderPath}/*.lstring")
    for path in tqdm(clusteredFiles[:]):
        convertLetter(path, combined_pickle_path)
    print("[2] completed to convert all lstring files!")

    print("[3] Now, extracting lstring files to orders")
    os.makedirs(tablePath, exist_ok=True)
    os.makedirs(orderPath, exist_ok=True)
    files = glob(f"./{convertedPath}/*.lstring")
    files = glob(f"./{clusteredFolderPath}/*.lstring")

    p = mp.Pool(mp.cpu_count())
    for path in files:
        p.apply_async(get_values, args=(path,))
    p.close()
    p.join()
    print("[3] Finished to extract the lstring information")

    print("[4] Started generate csv files for the seq2seq")
    tableFiles = glob(f"./{tablePath}/*.json")
    get_train_data(tableFiles)
    files = glob(f"{trainDataPath}/*")
    split_to_train_test(files)
    print(f"[4] Done! train and test csv files are at {csvFilePath}")

    seconds = time.time() - start_time
    print("Time Taken:", time.strftime("%H:%M:%S", time.gmtime(seconds)))


if __name__ == "__main__":
    tree_options = ["Acacia"]
    for t in tree_options:
        start_process(t)