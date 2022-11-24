python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

gunicorn --bin 0.0.0.0:5200 app:app
открывать как http://localhost:5200

deactivate
