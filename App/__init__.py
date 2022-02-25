from flask import Flask
import gunicorn

app = Flask(__name__)

from App import routes