from pickle import TRUE
import string
import random
import sys

# Insert an input seed to begin the calculation
while TRUE:
    seed = input("Enter an alphanumerical seed: ")

    # Encryption Algorithm - basic version
    encrypter = []

    # I tried to study the functions for understanding alphanumerical characters
    for i in range(0, len(seed)):
        for letter in seed:
            #if letter.isnumeric():
            encrypter.append(random.choice(list(string.ascii_letters)))
            #if letter.isalpha():
                #encrypter.append(random.choice(list(string.digits)))
            #if letter.isspace():
                #encrypter.append(random.choice(list(string.ascii_letters)))

        random.shuffle(encrypter)

    random.shuffle(encrypter)

    # Now due to the fact that the shuffle has increased the list size we need to reestablish the correct dimension of it,
    # based on the seed length. After that we can print the password.
    password = ""
    for i in range(0, int(len(encrypter)/len(seed))):
        password += str(encrypter[i])

    print("Generated password is: " + password)
    choice = input("continue? ... y/n ")

    if choice != 'y':
        sys.exit()
