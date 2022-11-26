import sys
sys.path.insert(0, '/Users/vbulash/Sites/face')

activate_this = '/Users/vbulash/.local/share/virtualenvs/facial_orientation-lV3X0h-7/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__ = activate_this))

from app import app as application