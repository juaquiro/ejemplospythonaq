import math

class MyClass:


    def __init__(self, name):
        self.name = name

    def method(self):
        return 'instance method called', self

    def say_hi(self):
        print('Hello my name is', self.name)

    @classmethod
    def classmethod(cls):
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'


class Pizza:


    #zona de instance methods

    # The __init__ method is similar to constructors in C++ and Java.
    # Constructors are used to initialize the object’s state.
    def __init__(self, radius, ingredients):
        self.ingredients = ingredients
        self.radius = radius

    # __repr__ is a special method used to represent a class’s objects as a string. _
    # _repr__ is called by the repr() built-in function.
    # You can define your own string representation of your class objects using the __repr__ method.
    def __repr__(self):
        return (f'Pizza({self.radius!r}, '
                f'{self.ingredients!r})')

    def area(self):
        return self.circle_area(self.radius)

    # zona de metodos estaticos
    @staticmethod
    def circle_area(r):
        return r ** 2 * math.pi


    #zona de class methods
    # class methods que vamos a usar como factories
    @classmethod
    def margherita(cls):
        radius=4
        return cls(radius, ['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        radius=6
        return cls(radius, ['mozzarella', 'tomatoes', 'ham'])



if __name__ == '__main__':
    r1=10
    r2=20
    p1=Pizza(r1, ['tomate', 'Cebolla'])
    a1=p1.area()
    # ejemplo de uso de f strings
    print(f"1st pizza {p1}, with radius {p1.radius} cm and area {p1.area()} cm^2")

    p2=Pizza.margherita()
    a2=p2.area()
    # ejemplo de uso de f strings
    print(f"2nd pizza {p2}, with radius {p2.radius} cm and area {p2.area()} cm^2")