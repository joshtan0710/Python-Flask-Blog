from flask import Flask, render_template, url_for  # import flask class
app = Flask(__name__)    #init app, what is __name__? name of the module 


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

if __name__ == '__main__':  # only True if we run this script directly
    app.run(debug=True)