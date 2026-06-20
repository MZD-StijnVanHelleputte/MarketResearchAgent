from fastapi import Depends


def get_agent():
    raise NotImplementedError


def get_checkpointer():
    raise NotImplementedError
