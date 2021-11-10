def add_or_subtract(num_1, num_2, subtract=False):
    """Add or subtract two numbers, depending on the value of subtract."""
    if subtract:
        return num_1 - num_2

    else:
        return num_1 + num_2

if __name__ == '__main__':
    #Functions Accept Positional and Keyword Arguments in Python

    #positional arguments
    assert add_or_subtract(10, 20, False)==30
    assert add_or_subtract(10, 20, True)==-10

    #keyword aguments, if all are keyword order does not matter
    assert add_or_subtract(num_1=10, num_2=20, subtract=True) ==add_or_subtract(subtract=True, num_2=20, num_1=10)

    #keyword and positional, first muy be the positional and the the keyword in any order
    assert add_or_subtract(10, 20, subtract=True)==-10


