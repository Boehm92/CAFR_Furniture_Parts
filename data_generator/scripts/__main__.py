import madcad as mdc
import pandas as pd
from cube_transform import *
import machining_feature_transform as mft

cad_directory = 'TRAINING_DATASET_SOURCE'

for i in range(1,  7168):
    print("Part: ", i)
    label_list = []

    add_chamfer = np.random.randint(0, 2)
    board_length = np.random.uniform(600, 2000)
    board_height = np.random.uniform(200, 600)
    wooden_board = mdc.brick(width=mdc.vec3(1))
    wooden_board = wooden_board.transform(mdc.mat3(board_length, 18, board_height))
    wooden_board = wooden_board.transform(mdc.vec3((board_length / 2), 9, (board_height / 2)))

    try:
        if add_chamfer == 1:
            print("machining_feature: 13")
            wooden_board = mft.MachiningFeature(13, wooden_board, board_length, board_height).apply_feature()
            label_list.append([0, 0, 0, 0, 0, 0, 13])

    except:
         print(" machining feature not feasible")

    number_machining_features = np.random.randint(0, 30)
    for count in range(number_machining_features):
        try:
            machining_feature = np.random.randint(0, 12)
            print("machining_feature: ", machining_feature)

            wooden_board = mft.MachiningFeature(machining_feature, wooden_board, board_length, board_height).\
                apply_feature()

            label_list.append([0, 0, 0, 0, 0, 0, machining_feature])
        except:
            print(" machining feature not feasible")

    # mdc.show([wooden_board])
    mdc.write(wooden_board, os.getenv(cad_directory) + "/" + str(i) + ".stl")
    labels = pd.DataFrame(label_list)
    labels.to_csv(os.getenv(cad_directory) + "/" + str(i) + ".csv",
                  header=False, index=False)

    del wooden_board
    del labels
