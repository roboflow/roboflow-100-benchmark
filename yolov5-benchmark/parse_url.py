
import argparse
import re

def parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--url", required=True, help="url")
    args = vars(ap.parse_args())

    # api_key = args["api_key"] 
    url = args["url"] 
   
    res = re.split("/+", url)
    res = res[2] + " " + res[3] + " " + res[4]

    with open('../attributes.txt', 'w') as f:
        f.write(res)
    return 



        
if __name__ == "__main__":
    parser()