from kivy.app import App    
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image


class ColoredLabel(Label):
    def __init__(self, color=(0, 0, 0, 1), text_color=(0, 0, 0, 1), **kw):
        super(ColoredLabel, self).__init__(**kw)
        self.color = text_color  # Set the text color
        with self.canvas.before:
            self.bg_color = Color(*color)  # Use the provided background color
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)

    def _update_bg(self, instance, value):
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos


class MainWindow(Screen):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.cols = 1

        self.window = FloatLayout(size_hint=(1, 1))
        with self.window.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(size=self.window.size, pos=self.window.pos)
            self.window.bind(size=self._update_rect, pos=self._update_rect)

        ###

        self.number = 1

        self.glass = Image(
            source = f"glasses/{self.number}.png",
            size_hint = (0.3, 0.3),
            pos_hint = {"x": 0, "top": 0.5},
        )
        self.window.add_widget(self.glass)

        self.plus_button = Button(
            background_normal = "glasses/plus.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.4, "top": 0.4},
            on_press = self.plus
        )
        self.window.add_widget(self.plus_button)
        
        self.minus_button = Button(
            background_normal = "glasses/minus.png",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.6, "top": 0.4},
            on_press = self.minus
        )
        self.window.add_widget(self.minus_button)

        self.label = ColoredLabel(
            text = f"{self.number}/8",
            size_hint = (0.1, 0.1),
            pos_hint = {"x": 0.8, "top": 0.4},
            color = (0, 0, 0, 1), 
            text_color = (1, 1, 1, 1),  
        )
        self.window.add_widget(self.label)

        ###

        self.add_widget(self.window)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def plus(self, instance):
        if (self.number < 8):
            self.number += 1
            self.glass.source = f"glasses/{self.number}.png"
            self.glass.reload()
            self.label.text = f"{self.number}/8"
        else:
            pass

    def minus(self, instance):
        if(self.number > 1):
            self.number -= 1
            self.glass.source = f"glasses/{self.number}.png"
            self.glass.reload(),
            self.label.text = f"{self.number}/8"
        else:
            pass

class WindowManager(ScreenManager):
    def __init__(self, **kw):
        super(WindowManager, self).__init__(**kw)

        self.add_widget(MainWindow(name='main'))

class MainApp(App):
    def build(self):
        return WindowManager()
    
if __name__ == '__main__':
    MainApp().run()