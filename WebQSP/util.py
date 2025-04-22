import json
import random
import re
import os
import math
import logging

# 定义允许的函数及其参数数量
allowed_functions = {
    "AND": 2,
    "COUNT": 1,
    "JOIN": 2,
    "ARGMAX": 2,
    "ARGMIN": 2,
    "LT": 2,
    "LE": 2,
    "GT": 2,
    "GE": 2,
    "TC": 3,
    "R": 1,
}
valid_masks = {"[ MASK_R ]", "[ MASK_E ]"}


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


def conv_func_upper(expression):
    keywords = ["[ MASK_R ]", "[ MASK_E ]"]
    function_list = list(allowed_functions.keys())
    keywords.extend(function_list)

    for keyword in keywords:
        expression = expression.replace(keyword.lower(), keyword)

    return expression

def add_spaces(expression):
    # 添加空格
    expression = expression.replace("(", " ( ").replace(")", " ) ")
    expression = expression.replace("[", " [ ").replace("]", " ] ")
    expression = expression.replace(",", " , ")

    # 去除多余空格
    expression = " ".join(expression.split())
    return expression


def check_balance(expression):
    stack = []
    # 遍历表达式中的每个字符
    for char in expression:
        if char == "(" or char == "[":
            stack.append(char)
        elif char == ")" or char == "]":
            if not stack:
                return False
            top = stack.pop()
            # 确保配对的括号类型相同
            if char == ")" and top != "(":
                return False
            if char == "]" and top != "[":
                return False
    return not stack  # 如果栈为空，说明括号平衡


# 检查是否为时间参数
def is_time_argument(arg):
    return arg.upper() == "NOW" or re.match(r"^\d+$", arg)


def split_arguments(args):
    result = []
    current = []
    balance = 0  # 用于跟踪小括号的平衡
    i = 0
    while i < len(args):
        if args[i] == "(":
            balance += 1
            current.append(args[i])
        elif args[i] == ")":
            balance -= 1
            current.append(args[i])
        elif args[i] == "[" and balance == 0:
            # 识别到方括号，读取直到匹配的闭括号
            current.append(args[i])
            i += 1
            while i < len(args) and args[i] != "]":
                current.append(args[i])
                i += 1
            current.append(args[i])  # 添加闭括号
            result.append("".join(current).strip())  # 将整个占位符作为一个参数
            current = []  # 清空 current
            i += 1  # 跳过闭括号
        elif args[i] == "\"" and balance == 0:
            # 处理带引号的字符串参数
            current.append(args[i])
            i += 1
            while i < len(args) and args[i] != "\"":
                current.append(args[i])
                i += 1
            if i < len(args):
                current.append(args[i])  # 添加闭引号
            result.append("".join(current).strip())
            current = []
        elif args[i] == " " and balance == 0:
            # 当前参数结束，只有在小括号平衡时才分割
            if current:
                result.append("".join(current).strip())
                current = []
        else:
            current.append(args[i])
        i += 1

    # 添加最后一个参数
    if current:
        result.append("".join(current).strip())
    return result


def validate_frame(expression):

    expression = expression.strip()
    if not expression:
        return False

    # 如果是合法的占位符，直接返回 True
    if expression.upper() in valid_masks:
        return True

    # 如果是引号括起的字符串，视为合法
    if expression.startswith("\"") and expression.endswith("\""):
        return True

    # 如果是合法的函数调用
    if expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()
        parts = expression.split(maxsplit=1)
        if len(parts) < 2:
            return False

        func, args = parts[0], parts[1]
        if func.upper() not in allowed_functions:
            return False
        func = func.upper()
        args = split_arguments(args)
        if len(args) != allowed_functions[func]:
            return False

        # 针对 TC 函数的特殊处理
        if func == "TC":
            return validate_frame(args[0]) and validate_frame(args[1]) and is_time_argument(args[2])

        # 递归检查每个参数是否合法
        return all(validate_frame(arg) for arg in args)

    return False


# 主函数：检查表达式是否合法
def is_valid_frame(expression):
    # 检查括号平衡
    if not check_balance(expression):
        return False
    # 验证结构
    return validate_frame(expression)



