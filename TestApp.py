import time
import threading


def task():
    print ("Start")
    time.sleep(5)
    print ("Stop")

def funcOne():
    print ("Func One")


def funcTwo():
    print ("Func Two")

def main():
    threading.Thread(target=task).start()
    threading.Thread(target=funcOne).start()
    threading.Thread(target=funcTwo).start()


main()