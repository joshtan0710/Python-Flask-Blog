from flask import Flask, render_template, url_for  # import flask class
from forms import Registration, LoginForm  # calling forms module that you have defined

app = Flask(__name__)    #init app, what is __name__? name of the module 


app.config['SECRET_KEY'] = '5b68021e391c826d777747d5b304fbfc'

posts=[
    {   
        'author': 'Joshua Tan',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted':'Jan 21 2021'
    },
    {   
        'author': 'Bob Tan',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted':'Jan 22 2021'
    }]




@app.route('/home')  #decorators, root page of our website
def hello_world():
    return render_template('home.html', posts = posts) # call template, get the posts data 

@app.route("/about")
def about():
    return render_template('about.html', title = 'About')


@app.route("/register")
def register():
    form = Registration()
    return render_template('register.html', title = 'Register', form = form )


@app.route("/login")
def login():
    login = LoginForm()
    return render_template('login.html', title = 'Register', form = form )

if __name__ == '__main__':  # only True if we run this script directly
    app.run(debug=True)