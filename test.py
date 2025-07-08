from pyremotedata.implicit_mount import IOHandler

if __name__ == "__main__":
    with IOHandler(verbose=True) as io:
        print(f'{io.ls()=}')