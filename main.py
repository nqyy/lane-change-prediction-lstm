from process_data import run
import pickle
import os

if not os.path.exists("output"):
    os.makedirs("output")

total_change = 0
for i in range(1, 61):
    number = "{0:0=2d}".format(i)
    result, change_num = run(number)
    total_change += change_num
    print("total changes:", total_change)

    filename = "output/result" + number + ".pickle"
    f = open(filename, 'wb')
    pickle.dump(result, f)
    print("Successfully write to:", filename)
    f.close()
