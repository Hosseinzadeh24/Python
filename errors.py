print("before")
try:
    prnt("oops")
except NameError as e:
    print("caught:", e)
