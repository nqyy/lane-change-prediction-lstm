from process_data import run
import pickle
import os

if not os.path.exists("output"):
    os.makedirs("output")

for i in range(1, 58):
    number = "{0:0=2d}".format(i)
    result = run(number)

    filename = "output/result" + number + ".pickle"
    f = open(filename, 'wb')
    pickle.dump(result, f)
    print("Successfully write to:", filename)
    f.close()
