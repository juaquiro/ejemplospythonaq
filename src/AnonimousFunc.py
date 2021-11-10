#Anonymous Functions Are Created With the lambda Keyword in Python
from scipy import integrate

say_helloFun = lambda x: print(f"Hello, {x:s}")

cuadFun = lambda x: x ** 2

if __name__ == '__main__':
    say_helloFun(f"Eleanor")

    v=integrate.quad(cuadFun, 0, 9)

    print(f"el resultado de la integral  de cuadFun es , {v}")






