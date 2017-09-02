class Human:
    def __init___(self, name, sex, age):
        self.name = name
        self.sex = sex
        self.age = age

    def say_hello(self):
        print("hello")

shintaro = Human("shintaro", 22, "man")
print(shintaro.say_hello())
