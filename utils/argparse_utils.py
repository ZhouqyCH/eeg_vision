import argparse
import os
from datetime import datetime


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def valid_path(p):
    if os.path.isdir(p):
        return p
    else:
        msg = "Not a valid path: '{0}'".format(p)
        raise argparse.ArgumentTypeError(msg)


def valid_output(file_name):
    failed = True
    msg = "File name '{0}' is invalid".format(file_name)
    if os.path.isdir(file_name):
        msg = "Expecting a file name but got a directory: '{0}'".format(file_name)
    elif os.path.isfile(file_name):
        msg = "File '{0}' already exist and would be overwritten".format(file_name)
    else:
        path = os.path.dirname(file_name)
        if valid_path(path):
            failed = False
    if failed:
        raise argparse.ArgumentTypeError(msg)
    return file_name


def valid_input(file_name):
    if os.path.isdir(file_name):
        msg = "Expecting a file name but got a directory: '{0}'".format(file_name)
        raise argparse.ArgumentTypeError(msg)

    if not os.path.isfile(file_name):
        msg = "File '{0}' does not exist.".format(file_name)
        raise argparse.ArgumentTypeError(msg)
    else:
        return file_name
