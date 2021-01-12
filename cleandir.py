import os

def main():
    filenames = [
        'interpolate.txt',
        'extrapolate.txt',
        'train-easy.txt',
        'train-medium.txt',
        'train-hard.txt',
        'train-all.txt'
    ]

    for file in filenames:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()