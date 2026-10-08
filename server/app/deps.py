from fastapi import Request

from .db import Database


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_meta(request: Request) -> dict:
    return request.app.state.meta
