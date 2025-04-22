import json
import random
import re
import os
import math
import logging


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
valid_masks = {"[ MASK_C ]", "[ MASK_R ]", "[ MASK_E ]"}


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

    expression = expression.replace("(", " ( ").replace(")", " ) ")
    expression = expression.replace("[", " [ ").replace("]", " ] ")
    expression = expression.replace(",", " , ")


    expression = " ".join(expression.split())
    return expression


def check_balance(expression):
    stack = []

    for char in expression:
        if char == "(" or char == "[":
            stack.append(char)
        elif char == ")" or char == "]":
            if not stack:
                return False
            top = stack.pop()

            if char == ")" and top != "(":
                return False
            if char == "]" and top != "[":
                return False
    return not stack



def is_time_argument(arg):
    return arg.upper() == "NOW" or re.match(r"^\d+$", arg)


def split_arguments(args):
    result = []
    current = []
    balance = 0
    i = 0
    while i < len(args):
        if args[i] == "(":
            balance += 1
            current.append(args[i])
        elif args[i] == ")":
            balance -= 1
            current.append(args[i])
        elif args[i] == "[" and balance == 0:

            current.append(args[i])
            i += 1
            while i < len(args) and args[i] != "]":
                current.append(args[i])
                i += 1
            current.append(args[i])
            result.append("".join(current).strip())
            current = []
            i += 1
        elif args[i] == "\"" and balance == 0:

            current.append(args[i])
            i += 1
            while i < len(args) and args[i] != "\"":
                current.append(args[i])
                i += 1
            if i < len(args):
                current.append(args[i])
            result.append("".join(current).strip())
            current = []
        elif args[i] == " " and balance == 0:

            if current:
                result.append("".join(current).strip())
                current = []
        else:
            current.append(args[i])
        i += 1


    if current:
        result.append("".join(current).strip())
    return result


def validate_frame(expression):

    expression = expression.strip()
    if not expression:
        return False


    if expression.upper() in valid_masks:
        return True


    if expression.startswith("\"") and expression.endswith("\""):
        return True


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


        if func == "TC":
            return validate_frame(args[0]) and validate_frame(args[1]) and is_time_argument(args[2])


        return all(validate_frame(arg) for arg in args)

    return False


def is_valid_frame(expression):

    if not check_balance(expression):
        return False

    return validate_frame(expression)



