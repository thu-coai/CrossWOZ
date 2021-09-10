import string
import random

from flask import Flask, session, g

from .models import *
from .routes import *

app = Flask(__name__)

app.register_blueprint(misc.bp, url_prefix='')
app.register_blueprint(room.bp, url_prefix='/room')
app.register_blueprint(services.bp, url_prefix='/services')
app.register_blueprint(match.bp, url_prefix='/match')

app.config.from_pyfile('settings.py')

invitation_code = '959592'

@app.before_request
def before_request():
    g.invitation_code = invitation_code

    user_id = session.get('user_id')
    try:
        g.me = User.get(User.id == user_id)
    except:
        g.me = None
