import os

def print0(content="", **kwargs):
    if os.environ.get("RANK",0) == 0:
        print(content)