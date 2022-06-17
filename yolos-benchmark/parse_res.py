import os
from os import path

def main():
    # read the file with stdout data
    res_file = open("yolos_res.txt", "r")
    # get list of lines
    lines = res_file.readlines()
    res_file.close()

    # grad the line with the value we are interested in
    res_file = open("yolos_res.txt", "w")
    for line in lines:
        if "IoU=0.50 " in line.strip("\n"):
            # Delete "line2" from new_file
            
            
            res_file.write(line)
            res = "YOLOs:" , line[-6:] # value that we want to save to res file

    res_file.close()
    
    if path.exists("../mAP_results.txt"):
        print("We should get in here")
        mAP_file = open("../mAP_results.txt", "r")
        # get list of lines
        lines = mAP_file.readlines()
        mAP_file.close()

        mAP_file = open("../mAP_results.txt", "w")
        for line in lines:
            if "YOLOs" not in line.strip("\n"):
                mAP_file.write(line)
        mAP_file.write("\n")
        mAP_file.write(''.join(res))

        mAP_file.close()

    else:
    
        with open('../mAP_results.txt', 'w') as f:
            f.write(''.join(res))





if __name__ == "__main__":
    main()