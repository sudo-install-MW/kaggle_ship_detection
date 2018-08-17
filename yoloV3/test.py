# def a(b):
#     print("start")
#     c = b()
#     print("end")

# def b():
#     print("exec b")

# print(a(b))

# def add(a):
#     def add2(b):
#         print(a+b)
#     return add2

# add3 = add(3)
# add4 = add(4)

# add3(4)
# add4(8)

# def my_decorator(some_function):

#     def wrapper():

#         num = 10

#         if num == 10:
#             print("Yes!")
#         else:
#             print("No!")

#         some_function()

#         print("Something is happening after some_function() is called.")

#     return wrapper


# @my_decorator
# def just_some_function():
#     print("Wheee!")

# just_some_function()
