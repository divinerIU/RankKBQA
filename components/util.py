import pickle
import json
import os
import shutil
import re
from typing import List
from executor.sparql_executor import get_label_with_odbc


def dump_to_bin(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w', encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)


illegal_xml_re = re.compile(u'[\x00-\x08\x0b-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff\ufdd0-\ufddf\ufffe-\uffff]')


def clean_str(s: str) -> str:
    """remove illegal unicode characters"""
    return illegal_xml_re.sub('', s)


def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    return toks


def extract_mentioned_entities_from_sexpr(expr: str) -> List[str]:
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    entitiy_tokens = []
    for t in toks:
        # normalize entity
        if t.startswith('m.') or t.startswith('g.'):
            entitiy_tokens.append(t)
    return entitiy_tokens


def extract_mentioned_entities_from_sparql(sparql: str) -> List[str]:
    """extract entity from sparql"""
    sparql = sparql.replace('(', ' ( ').replace(')', ' ) ')
    toks = sparql.split(' ')
    toks = [x.replace('\t.', '') for x in toks if len(x)]
    entity_tokens = []
    for t in toks:
        if t.startswith('ns:m.') or t.startswith('ns:g.'):
            entity_tokens.append(t[3:])

    entity_tokens = list(set(entity_tokens))
    return entity_tokens


def extract_mentioned_relations_from_sparql(sparql: str):
    """extract relation from sparql"""
    sparql = sparql.replace('(', ' ( ').replace(')', ' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []
    for t in toks:
        if (re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*", t.strip())
                or re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*", t.strip())):
            relation_tokens.append(t[3:])

    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def extract_mentioned_entities_from_graph_query(graph_query):
    """extract entity from graph_query"""
    node_list = graph_query["nodes"]
    entity_tokens = []
    if node_list:
        for node in node_list:
            if node["node_type"] == "entity":
                entity_tokens.append(node['id'])
    entity_tokens = list(set(entity_tokens))
    return entity_tokens


def extract_mentioned_relations_from_graph_query(graph_query):
    """extract relation from graph_query"""
    relation_list = graph_query["edges"]
    relation_tokens = []
    if relation_list:
        for relation in relation_list:
            relation_tokens.append(relation['relation'])
    relation_tokens = list(set(relation_tokens))

    return relation_tokens


def extract_mentioned_class_from_graph_query(graph_query):
    """extract relation from graph_query"""
    node_list = graph_query["nodes"]
    class_tokens = []
    if node_list:
        for node in node_list:
            if node["node_type"] == "class":
                class_tokens.append(node['id'])
    class_tokens = list(set(class_tokens))
    return class_tokens


def extract_mentioned_relations_from_sexpr(sexpr: str) -> List[str]:
    sexpr = sexpr.replace('(', ' ( ').replace(')', ' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []

    for t in toks:
        if (re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-z_]*", t.strip())
                or re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*", t.strip())):
            relation_tokens.append(t)
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def vanilla_sexpr_linearization_method(expr, entity_label_map={}, relation_label_map={}, linear_origin_map={}):
    """
    textualize a logical form, replace mids with labels

    Returns:
        (str): normalized s_expr
    """
    expr = expr.replace("(", " ( ")  # add space for parantheses
    expr = expr.replace(")", " ) ")
    toks = expr.split(" ")  # split by space
    toks = [x for x in toks if len(x)]

    norm_toks = []
    for t in toks:

        # original token
        origin_t = t

        if t.startswith("m.") or t.startswith("g."):  # replace entity with its name
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                # name = get_label(t)
                name = get_label_with_odbc(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
            t = '[ ' + t + ' ]'
        elif "XMLSchema" in t:  # remove xml type
            format_pos = t.find("^^")
            t = t[:format_pos]
        elif t == "ge":  # replace ge/gt/le/lt
            t = "GREATER EQUAL"
        elif t == "gt":
            t = "GREATER THAN"
        elif t == "le":
            t = "LESS EQUAL"
        elif t == "lt":
            t = "LESS THAN"
        else:
            t = t.replace("_", " ")  # replace "_" with " "
            t = t.replace(".", " , ")  # replace "." with " , "

            if "." in origin_t:  # relation
                t = "[ " + t + " ]"
                relation_label_map[origin_t] = t

        norm_toks.append(t)
        linear_origin_map[t] = origin_t  # for reverse transduction

    return " ".join(norm_toks)


def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r:  # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r:  # replace "." with " , "
        r = r.replace(".", " , ")
    return r


def add_spaces(expression):
    # 添加空格
    expression = expression.replace("(", " ( ").replace(")", " ) ")
    expression = expression.replace("[", " [ ").replace("]", " ] ")

    expression = expression.replace(",", " , ")

    # 去除多余空格
    expression = " ".join(expression.split())
    return expression


# 检查括号是否平衡
def check_parentheses_balance(expression):
    stack = []
    for char in expression:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return False
            stack.pop()
    return not stack


# 分割参数，处理嵌套结构和方括号
def split_arguments(args):
    result = []
    current = []
    balance = 0  # 小括号平衡
    square_balance = 0  # 方括号平衡
    i = 0
    while i < len(args):
        if args[i] == "(":
            balance += 1
            current.append(args[i])
        elif args[i] == ")":
            balance -= 1
            current.append(args[i])
        elif args[i] == "[":
            square_balance += 1
            current.append(args[i])
        elif args[i] == "]":
            square_balance -= 1
            current.append(args[i])
        elif args[i] == " " and balance == 0 and square_balance == 0:
            if current:
                result.append("".join(current).strip())
                current = []
        else:
            current.append(args[i])
        i += 1
    if current:
        result.append("".join(current).strip())
    return result


# 检查是否为时间参数
def is_time_argument(arg):
    return arg == "NOW" or re.match(r"^\d{4}$", arg)


# 验证表达式结构是否合法
def validate_structure(expression):
    # 定义允许的函数及其参数数量
    allowed_functions = {
        "AND": 2,
        "COUNT": 1,
        "R": 1,
        "JOIN": 2,
        "ARGMAX": 2,
        "ARGMIN": 2,
        "LT": 2,
        "LE": 2,
        "GT": 2,
        "GE": 2,
        "TC": 3,
    }
    # 去掉首尾空格
    expression = expression.strip()
    if not expression:
        return False

    # 检查是否为合法占位符或属性
    if re.match(r"^[\w\.\[\], ]+$", expression):  # 匹配简单变量或属性列表
        return True

    # 如果是合法的函数调用
    if expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()  # 去掉最外层括号
        parts = expression.split(maxsplit=1)
        if len(parts) < 2:
            return False  # 函数名和参数至少需要两个部分

        func, args = parts[0], parts[1]
        if func not in allowed_functions:
            return False  # 函数名非法
        # 根据参数数量分割参数
        args = split_arguments(args)
        if len(args) != allowed_functions[func]:
            return False  # 参数数量不匹配

        # 针对 TC 函数的特殊处理
        if func == "TC":
            # 第一个参数是普通表达式，第二个参数必须满足 `[属性列表] 时间参数` 格式
            return validate_structure(args[0]) and validate_structure(args[1]) and is_time_argument(args[2])

        # 递归验证每个参数
        return all(validate_structure(arg) for arg in args)

    # 默认非法
    return False


# 主函数：验证表达式是否合法
def is_valid_expression(expression):
    if not check_parentheses_balance(expression):
        return False  # 括号不平衡
    return validate_structure(expression)


def filter_mask_patterns(predicts):
    filter_pre = []
    for p in predicts:
        predict = add_spaces(p[0])
        if is_valid_expression(predict):
            filter_pre.append(predict)

    return filter_pre
