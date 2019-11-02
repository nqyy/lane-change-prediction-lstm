from process_data import run
import pickle

for i in range(1, 61):
    if i < 10:
        number = "0" + str(i)
    else:
        number = str(i)
    result = run(number)

    f = open("output/result" + number + ".pickle", 'wb')
    pickle.dump(result, f)
    print("Successfully write to lane changing pickle file:", number)
    f.close()
