import joblib
from keras.preprocessing import image
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from sklearn import svm
import pickle


# Painting Board Subwidget:
# Define a widget and the corresponding UI operations for painting
class PaintWidget(Widget):
    color = (254, 254, 254, 1)  # Pen color
    thick = 13  # Pen thickness

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.parent_widget = root

    def on_touch_down(self, touch):
        with self.canvas:
            Color(*self.color, mode='rgba')
            if touch.x > self.width or touch.y < self.parent_widget.height - self.height:
                return
            touch.ud['line'] = Line(
                points=(touch.x, touch.y), width=self.thick)

    def on_touch_move(self, touch):
        with self.canvas:
            if touch.x > self.width or touch.y < self.parent_widget.height - self.height:
                return
            touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        if touch.x > self.width or touch.y < self.parent_widget.height - self.height:
            return
        self.export_to_png('r.png')
        self.parent.do_predictions()


# Recognizer
# Define the application window, and some corresponding operations
class Recognizer(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

  # Initialize the CNN model from the trained model

        self.predictor = SVMPredictor()

        self.number = -1  # Variable to store the predicted number
        self.orientation = 'horizontal'  # UI related  UI相关
        self.draw_window()

    # function to declare the components of the application, and add them to the window
    def draw_window(self):
        # Clear button
        self.clear_button = Button(text='Cancel', font_name=SVM_Handwritten_RecognizerApp.font_name, size_hint=(1, 4 / 45),
                                   background_color=(255, 165 / 255, 0, 1))
        # Painting board
        self.painter = PaintWidget(self, size_hint=(1, 8 / 9))
        # Label for hint text
        self.hint_label = Label(
            font_name=SVM_Handwritten_RecognizerApp.font_name, size_hint=(1, 1 / 45))
        # Label for predicted number
        self.result_label = Label(font_size=200, size_hint=(1, 1 / 3))

        # BoxLayout
        first_column = BoxLayout(orientation='vertical', size_hint=(2 / 3, 1))
        second_column = BoxLayout(orientation='vertical', size_hint=(1 / 3, 1))
        # Add widgets to the window
        first_column.add_widget(self.painter)
        first_column.add_widget(self.hint_label)
        second_column.add_widget(self.result_label)
        second_column.add_widget(self.clear_button)
        self.add_widget(first_column)
        self.add_widget(second_column)

        # motion binding
        # Bind the click of the clear button to the clear_paint function

        self.clear_button.bind(on_release=self.clear_paint)

        self.clear_paint()  # Initialize the state of the app  初始化应用状态

    # Clear the painting board and initialize the state of the app.
    def clear_paint(self, obj=None):
        self.painter.canvas.clear()
        self.number = -1
        self.result_label.text = '^-^'
        self.hint_label.text = 'Please draw a digit on the board~'
        self.info_board.text = 'Info Board'

    # Extract info from the predictions, and display them on the window
    def show_info(self, predictions):
        self.result_label.text = predictions
        self.hint_label.text = 'The predicted character is displayed.'

    # Use CNN predictor to do prediction, and call show_info to display the result
    def do_predictions(self):
        pre = self.predictor.get_predictions('r.png')
        self.show_info(pre)


# Main app class
class SVM_Handwritten_RecognizerApp(App):
    font_name = r'Calibri.ttf'

    def build(self):
        return Recognizer()


if __name__ == '__main__':
    SVM_Handwritten_RecognizerApp().run()


class SVMPredictor():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_result(result):

        if result == [0]:
            return('A')
        elif result == [1]:
            return ('B')
        elif result == [2]:
            return ('C')
        elif result == [3]:
            return ('D')
        elif result == [4]:
            return ('E')
        elif result == [5]:
            return ('F')
        elif result == [6]:
            return ('G')
        elif result == [7]:
            return ('H')
        elif result == [8]:
            return ('I')
        elif result == [9]:
            return ('J')
        elif result == [10]:
            return ('K')
        elif result == [11]:
            return ('L')
        elif result == [12]:
            return ('M')
        elif result == [13]:
            return ('N')
        elif result == [14]:
            return ('O')
        elif result == [15]:
            return ('P')
        elif result == [16]:
            return ('Q')
        elif result == [17]:
            return ('R')
        elif result == [18]:
            return ('S')
        elif result == [19]:
            return ('T')
        elif result == [20]:
            return ('U')
        elif result == [21]:
            return ('V')
        elif result == [22]:
            return ('W')
        elif result == [23]:
            return ('X')
        elif result == [24]:
            return ('Y')
        elif result == [25]:
            return ('Z')
        elif result == [26]:
            return ('a')
        elif result == [27]:
            return ('b')
        elif result == [28]:
            return ('c')
        elif result == [29]:
            return ('d')
        elif result == [30]:
            return ('e')
        elif result == [31]:
            return ('f')
        elif result == [32]:
            return ('g')
        elif result == [33]:
            return ('h')
        elif result == [34]:
            return ('i')
        elif result == [35]:
            return ('j')
        elif result == [36]:
            return ('k')
        elif result == [37]:
            return ('l')
        elif result == [38]:
            return ('m')
        elif result == [39]:
            return ('n')
        elif result == [40]:
            return ('o')
        elif result == [41]:
            return ('p')
        elif result == [42]:
            return ('q')
        elif result == [43]:
            return ('r')
        elif result == [44]:
            return ('s')
        elif result == [45]:
            return ('t')
        elif result == [46]:
            return ('u')
        elif result == [47]:
            return ('v')
        elif result == [48]:
            return ('w')
        elif result == [49]:
            return ('x')
        elif result == [50]:
            return ('y')
        elif result == [51]:
            return ('z')

    def get_predictions(self, filename):

        dimension = (64, 75, 1)

        model = joblib.load(open("train_rbf_SVM.pkl", "rb"))

        pca = joblib.load(open("train_pca.pkl", "rb"))

        filename = r'C:/Users/Timilehin Vincent/Desktop/Desktop/project/SVM ONLINE RECOGNITION DATASET/training/B/icr_003_B_2_1.bmp'

        flat_data = []

        img = imread(filename)

        img_resized = resize(
            img, dimension, anti_aliasing=True, mode='reflect')

        flat_data.append(img_resized.flatten())

        flat_data = np.array(flat_data)

        test_img = pca.transform(flat_data)

        print('Image loaded------------')

        result = model.predict(test_img)

        result = self.get_result(result)

        return result
