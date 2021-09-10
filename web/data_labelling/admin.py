from dotenv import load_dotenv
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.peewee import ModelView
from flask_admin.contrib.fileadmin import FileAdmin
from flask import g, redirect, url_for
import os

from .models import *


class MyIndexView(AdminIndexView):
    def is_accessible(self):
        return g.me and g.me.is_admin

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('misc_bp.login'))


class MyFileAdmin(FileAdmin):
    def is_accessible(self):
        return g.me and g.me.is_admin

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('misc_bp.login'))


class MyModelView(ModelView):
    def is_accessible(self):
        return g.me and g.me.is_admin

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('misc_bp.login'))


class UserView(MyModelView):
    column_exclude_list = ['password_hash', ]


class TaskView(MyModelView):
    pass


class RoomView(MyModelView):
    pass


load_dotenv()

admin = Admin(
    name='任务导向对话系统· 管理界面',
    template_mode='bootstrap3',
    index_view=MyIndexView()
)

admin.add_views(
    UserView(User),
    TaskView(Task),
    RoomView(Room)
)

results_dir = os.path.join(os.path.dirname(__file__), 'results')
admin.add_view(MyFileAdmin(results_dir, name='Result Files'))
