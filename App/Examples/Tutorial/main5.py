from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

class MainWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

# kv = Builder.load_file("main5.kv")


class Main5App(App):
    def build(self):
        return # kv
    
if __name__ == "__main__":
    Main5App().run()