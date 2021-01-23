from flask import Flask, render_template, url_for, flash, redirect  # import flask class
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



@app.route('/')
@app.route('/home')  #decorators, root page of our website
def home():
    return render_template('home.html', posts = posts) # call template, get the posts data 

@app.route("/about")
def about():
    return render_template('about.html', title = 'About')


@app.route("/register", methods = ['GET', 'POST'])
def register():
    form = Registration()
    if form.validate_on_submit(): # if True
        flash(f'Account created for {form.username.data}!', 'success') #form.username.data returns data passed into it 
        return redirect(url_for('home'))  #redirect to function that stores homepage
    return render_template('register.html', title = 'Register', form = form )


@app.route("/login", methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == "admin@blog.com" and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful, check user and password', 'danger')
    return render_template('login.html', title = 'Register', form = form )

if __name__ == '__main__':  # only True if we run this script directly
    app.run(debug=True)