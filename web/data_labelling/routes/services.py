import re
import json
import bcrypt

from flask import Blueprint, render_template, redirect, request, session

from ..utils import *
from ..models import *

bp = Blueprint('services_bp', __name__)


@bp.route('/resetdb', methods=['POST'])
@admin_required
def service_resetdb():
    resetdb()

    return 'OK'


@bp.route('/register', methods=['POST'])
def service_register():
    data = request.get_data()
    try:
        data = json.loads(data)
        username = data['username']
        password = data['password']
        invitation_code = data['invitationCode']
    except:
        return '格式错误', 400

    if invitation_code != g.invitation_code:
        return '邀请码不正确', 403

    if not re.fullmatch(r'\w+', username):
        return '用户名仅含字母数字下划线', 400
    if not (2 <= len(username) <= 16):
        return '用户名的长度范围 [2, 16]', 400
    if not (8 <= len(password) <= 32):
        return '密码的长度范围 [8, 32]', 400
    if User.select().where(User.username == username):
        return '用户名已被使用', 422

    User.create(username=username, password_hash=bcrypt.hashpw(str.encode(password), bcrypt.gensalt()))

    return 'OK'


@bp.route('/login', methods=['POST'])
def services_login():
    data = request.get_data()
    try:
        data = json.loads(data)
        username = data['username']
        password = data['password']
    except:
        return '格式错误', 400

    try:
        user = User.get(User.username == username)
        assert bcrypt.checkpw(str.encode(password), str.encode(user.password_hash))
    except:
        return '用户名或密码错误', 401

    session['user_id'] = user.id
    session.permanent = True

    return 'OK'


@bp.route('/logout', methods=['POST'])
@login_required
def service_logout():
    session.pop('user_id')

    return 'OK'
