# *args and **kwargs in Python
#
#     Difficulty Level : Easy
#     Last Updated : 15 Dec, 2021
#
# In Python, we can pass a variable number of arguments to a function using special symbols. There are two special symbols:
# ver https://www.geeksforgeeks.org/args-kwargs-python/

# Python program to illustrate
# *args for variable number of arguments
def myFun1(*argv):
    for arg in argv:
        print (arg)
# Python program to illustrate
# *args with first extra argument
def myFun2(arg1, *argv):
    print ("First argument :", arg1)
    for arg in argv:
        print("Next argument through *argv :", arg)


# Python program to illustrate
# *kwargs for variable number of keyword arguments
def myFun3(**kwargs):
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))


# Python program to illustrate  **kwargs for
# variable number of keyword arguments with
# one extra argument.
def myFun4(arg1, **kwargs):
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))

# Using *args and **kwargs to call a function
def myFun5(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

# Using *args and **kwargs in same line to call a function
def myFun6(*args,**kwargs):
    print("args: ", args)
    print("kwargs: ", kwargs)


if __name__ == '__main__':
    funObjectList = (myFun1, myFun2, myFun3, myFun4, myFun5, myFun6)

    funObject=funObjectList[0]
    print(f'funcion: {funObject.__name__}')
    funObject('Hello', 'Welcome', 'to', 'GeeksforGeeks')
    print("\n")

    funObject = funObjectList[1]
    print(f'funcion: {funObject.__name__}')
    funObject('Hello', 'Welcome', 'to', 'GeeksforGeeks')
    print("\n")

    funObject = funObjectList[2]
    print(f'funcion: {funObject.__name__}')
    funObject(first = 'Geeks', mid ='for', last='Geeks')
    print("\n")

    funObject = funObjectList[3]
    print(f'funcion: {funObject.__name__}')
    funObject("Hi", first ='Geeks', mid ='for', last='Geeks')
    print("\n")

    # Now we can use *args or **kwargs to
    # pass arguments to this function :
    funObject = funObjectList[4]
    print(f'funcion: {funObject.__name__} using *args')
    args = ("Geeks", "for", "Geeks")
    funObject(*args)
    print(f'funcion: {funObject.__name__} using **kwargs')
    kwargs = {"arg1": "Geeks", "arg2": "for", "arg3": "Geeks"}
    funObject(**kwargs)
    print("\n")

    # Now we can use both *args ,**kwargs
    # to pass arguments to this function :
    funObject = funObjectList[5]
    print(f'funcion: {funObject.__name__} using *args and **kwargs')
    funObject('geeks','for','geeks',first="Geeks",mid="for",last="Geeks")