from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

class SayHello(App):
    def build(self):
        #returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}
        # changing the background colour
        Window.clearcolor = get_color_from_hex('#de140d')
        

        # image widget
        self.window.add_widget(Image(source="logo.png"))

        # label widget
        self.greeting = Label(
                        text= "What's your name?",
                        font_size= 18,
                        color= '##000105'
                        )
        self.window.add_widget(self.greeting)

        # text input widget
        self.user = TextInput(
                    multiline= False,
                    padding= (0, 20, 0, 20),
                    size_hint= (10, 0.4)
                    )

        self.window.add_widget(self.user)

        # button widget
        self.button = Button(
                      text= "GREET",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#000000',
                      #remove darker overlay of background colour
                      # background_normal = ""
                      )
        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        return self.window

    def callback(self, instance):
        # change label text to "Hello + user name!"
        self.greeting.text = "Hello " + self.user.text + "!"

# run Say Hello App Calss
if __name__ == "__main__":
    SayHello().run()