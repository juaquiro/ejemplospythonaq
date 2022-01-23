# Anonymous Functions Are Created With the lambda Keyword in Python
# de https://www.geeksforgeeks.org/python-lambda/?ref=lbp
# In Python, anonymous function means that a function is without a name.
# As we already know that def keyword is used to define the normal functions
# and the lambda keyword is used to create anonymous functions

from scipy import integrate

say_helloFun = lambda x: print(f"Hello, {x:s}")

cuadFun = lambda x: x ** 2

if __name__ == '__main__':

    varName= [ i for i, a in locals().items() if a == say_helloFun][0]
    print(f"anonynous funcion {varName}, {say_helloFun.__name__}\n")
    say_helloFun(f"Eleanor")

    varName = [i for i, a in locals().items() if a == cuadFun][0]
    print(f"anonynous funcion {varName}, {cuadFun.__name__}\n")
    v=integrate.quad(cuadFun, 0, 9)
    print(f"el resultado de la integral  de {varName} es , {v}\n")

    # Python program to demonstrate
    # lambda functions
    string = 'GeeksforGeeks'
    # lambda returns a function object
    print("lambda returns a function object: ")
    print(lambda string: string  + "\n")

    x = "GeeksforGeeks"

    # lambda gets pass to print
    print("lambda gets pass to print\n")
    (lambda x: print(x))(x)

    # Python program to illustrate cube of a number
    # showing difference between def() and lambda().
    def cube(y):
        return y * y * y;
    g = lambda x: x * x * x
    print(f"labda cube {g(7)}")
    print(f"def cube {cube(7)}")

    # Example  # 4: The lambda function gets more helpful when used inside
    # a function.

    print("lambda inside a function \n")
    def power(n):
        return lambda a: a ** n


    # base = lambda a : a**2 get
    # returned to base
    base = power(2)

    print("Now power is set to 2")

    # when calling base it gets
    # executed with already set with 2
    print("8 powerof 2 = ", base(8))

    # base = lambda a : a**5 get
    # returned to base
    base = power(5)
    print("Now power is set to 5")

    # when calling base it gets executed
    # with already set with newly 2
    print("8 powerof 5 = ", base(8))

    # Python program to demonstrate
    # lambda functions inside map()
    # and filter()
    print("lambda inside map and filter")

    a = [100, 2, 8, 60, 5, 4, 3, 31, 10, 11]

    # in filter either we use assignment or
    # conditional operator, the pass actual
    # parameter will get return
    print(f"list of numbers {a}")
    print("filter even numbers")
    filtered = filter(lambda x: x % 2 == 0, a)
    print(list(filtered))

    # in map either we use assignment or
    # conditional operator, the result of
    # the value will get returned
    print("map even numbers")
    mapped = map(lambda x: x % 2 == 0, a)
    print(list(mapped))








