'''
SCRIPT USAGE

python stripString.py . ../../working_fluoroSpine/FRANK_TEST/images/grey_image_6.jpg
../../working_fluoroSpine/FRANK_TEST/images/grey_image_6.jpg
['..', '..', 'working_fluoroSpine', 'FRANK_TEST', 'images', 'grey_image_6.jpg']
../../working_fluoroSpine/FRANK_TEST/images/

'''

import sys
path=sys.argv[2]
pathSplit = path.split("/")
print path
print pathSplit

finalpath=''
for pathSubPart in range(len(pathSplit)-1):
    finalpath+=pathSplit[pathSubPart]+'/'

print finalpath
