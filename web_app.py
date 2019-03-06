from flask import Flask, request, render_template
from face_recg import rcgn, prepare_img, prepare_img_rcgn


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def indx():
    return render_template('main.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    req = request.files.getlist('ex')

    images, photo_str = prepare_img(req)

    return """<!DOCTYPE html>
<html>
<head>

<style>
#hide_text {
  display: none;
}
</style>
</head>

<body>
<h2>Sloboda face-recognition example</h2>
<p>detection results</p>


<p>""" + images + '''</p>
<p>We found some faces and now we need several photo of man who you want to recognize</p>
<p>it's necessary, the only one man have to be present on photo</p>


<form action="/process_rcgn" method="post" enctype="multipart/form-data">
    <input type="text" id="hide_text" name="hide_text"  value=''' + photo_str + '''>

    <br><br>
    <input type="text" id="a" name="a"  style='width:50em' value='' placeholder="enter alias name this person">
    <br><br>
    <input type="file" name="ex" multiple>
    <br><br>
    <input type="submit" value="recognize">
</form>

</body>
</html>'''


@app.route('/process_rcgn', methods=['GET', 'POST'])
def process_rcgn():
    req = request.files.getlist('ex')
    text = request.form['a']
    hide_img = request.form['hide_text']
    print(hide_img)

    rcg = prepare_img_rcgn(req, hide_img, [text])

    return """<!DOCTYPE html>
    <html>
    <body>
    <h2>Sloboda face-recognition example</h2>

    <p>Recognize result</p>


    <p>""" + rcg + """</p>
    <form action="/" method="post">

        <input type="submit" value='restart'>
    </form>
    </body>
    </html>"""


if __name__ == '__main__':
    app.run()
