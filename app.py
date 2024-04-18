from flask import Flask

app = Flask(__name__)

#define routes
@app.route('/')
def index():
    return 'Flask Backend!'

@app.route('/api')
def api():
    return 'API endpoint'

if __name__== '__main__':
    app.run(debug=True)