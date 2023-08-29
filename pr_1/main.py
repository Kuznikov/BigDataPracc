# -*-coding: utf-8 -*-
import math


def one():
    print("Enter the name of the shape")
    tip = raw_input("Available shapes triangle, rectangle, circle = ")

    if tip == "triangle":
        a = float(input("Enter a = "))
        b = float(input("Enter b = "))
        c = float(input("Enter c = "))
        p = (a + b + c) / 2
        s = math.sqrt((p * (p - a) * (p - b) * (p - c)))
        print("Triangle:", s)
    elif tip == "rectangle":
        a = float(input("Enter a = "))
        b = float(input("Enter b = "))
        s = a * b
        print("Rectangle:", s)
    elif tip == "circle":
        r = float(input("Enter r = "))
        s = math.pi * (r ** 2)
        print("Circle:", s)
    else:
        print("Inappropriate value.")



def two():
    print("Ноль в качестве знака операции"
          "\nзавершит работу программы")
    while True:
        s = raw_input("Знак (+,-,*,/): ")
        if s == '0':
            break
        if s in ('+', '-', '*', '/'):
            x = float(input("x="))
            y = float(input("y="))
            if s == '+':
                print("%.2f" % (x + y))
            elif s == '-':
                print("%.2f" % (x - y))
            elif s == '*':
                print("%.2f" % (x * y))
            elif s == '/':
                if y != 0:
                    print("%.2f" % (x / y))
                else:
                    print("Деление на ноль!")
        else:
            print("Неверный знак операции!")


def three():
    a = int(raw_input())
    b = int(raw_input())
    c = int(raw_input())
    p = (a + b + c) / 2
    s = (p * (p - a) * (p - b) * (p - c)) ** 0.5

    print(s)

two()
