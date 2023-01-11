from flask import Flask, render_template, request, session
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import keras, os
import numpy as np


my_model = keras.models.load_model("plantv2.h5")
classification_name=['Alstonia Scholaris Diseased','Alstonia Scholaris healthy','Apple scab','Apple Black rot','Apple Cedar apple rust','Apple healthy','Arjun diseased','Arjun healthy','Bael diseased','Basil healthy','Blueberry healthy','Cherry (including sour) Powdery mildew','Cherry (including sour) healthy','Chinar diseased','Chinar healthy','Corn (maize) Cercospora Gray leaf spot','Corn (maize) Common rust','Corn (maize) Northern Leaf Blight','Corn (maize) healthy','Grape Black_rot','Grape Esca (Black Measles)','Grape Leaf blight (Isariopsis Leaf Spot)','Grape healthy','Guava diseased','Guava healthy','Jamun diseased','Jamun healthy','Jatropha diseased','Jatropha healthy','Lemon diseased','Lemon healthy','Mango diseased','Mango healthy','Orange Haunglongbing(Citrus greening)','Peach Bacterial spot','Peach healthy','Pepper bell Bacterial spot','Pepper bell healthy','Pomegranate diseased','Pomegranate healthy','Pongamia Pinnata diseased','Pongamia Pinnata healthy','Potato Early blight','Potato Late blight','Potato healthy','Raspberry healthy','Soybean healthy','Squash Powdery mildew','Strawberry Leaf scorch','Strawberry healthy','Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites Two-spotted spider mite','Tomato Target Spot','Tomato Yellow Leaf Curl Virus','Tomato mosaic virus','Tomato healthy']
treat=[0, 0, 't2.jpg', 't3.jpg', 't4.jpg', 0, 0, 0, 0, 0, 0, 't11.jpg', 0, 0, 0, 't15.jpg', 't16.jpg', 't17.jpg', 0, 't19.jpg', 't20.jpg', 't21.jpg', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 't34.jpg', 0, 't36.jpg', 0, 0, 0, 0, 0, 't42.jpg', 't43.jpg', 0, 0, 0, 't47.jpg', 't48.jpg', 0, 't50.jpg', 't51.jpg', 't52.jpg', 't53.jpg', 't54.jpg', 't55.jpg', 't56.jpg', 0, 't58.jpg', 0]


app = Flask(__name__,  template_folder='templates', static_folder='static')

IMG_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    a=0
    b=1
    return render_template('index.html')

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        img_filename = secure_filename(f.filename)
        # Upload file to database (defined uploaded folder in static path)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        leaf_img = session.get('uploaded_img_file_path', None)
        print(leaf_img)
        img = load_img(leaf_img, target_size=(256, 256))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = my_model.predict(images,batch_size=1)
        a=max(classes[0])
        print("max:",a)
        indx=0
        for i in classes[0]:
            if a==i:
                nind=indx
            indx+=1
        print(classification_name[nind])
        if treat[nind]!=0:
            print("treatment Available---")
            treatment="static/treatment/"+str(treat[nind])
            notreatment=""
        else:
            notreatment=""
            treatment="static/treatment/trar.JPG"
        return render_template('result.html',image=leaf_img, res=classification_name[nind],confident=int(a*100),treatment=treatment)

if __name__ == '__main__':
    app.run(debug=True)
