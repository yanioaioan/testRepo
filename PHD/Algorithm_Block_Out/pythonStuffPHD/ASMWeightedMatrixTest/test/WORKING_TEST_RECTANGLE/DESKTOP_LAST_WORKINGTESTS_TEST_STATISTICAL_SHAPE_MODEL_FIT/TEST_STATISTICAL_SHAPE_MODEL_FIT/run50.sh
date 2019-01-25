#for i in {100..150}; do python segmentImage.py phdtest/points/1-100 phdtest/images/50-150/grey_image_"$((0+i))".jpg ; done;
#for i in {101..125}; do python segmentImage.py phdtest/points/ANDERST-ADAM/C5/1-50 phdtest/images/ANDERST-ADAM/C5/51-100-havent_marked_yet/51-75/grey_image_"$((i-100))".jpg ; done; 

#from 50 to 90 frames
#for i in {1..41}; do  python segmentImage.py phdtest/points/ANDERST-ADAM/C5/1-96/enhanced/50-90 phdtest/images/ANDERST-ADAM/C5/1-96/enhanced/inverted/50-90/targetFlex2Iinline/grey_image_"$((i))".jpg; done;

for i in {1..25}; do  python segmentImage.py phdtest/points/lastworking/1-50 phdtest/images/lastworking/1-50/target1-100_first_1_25/grey_image_"$((i))".jpg; done;




