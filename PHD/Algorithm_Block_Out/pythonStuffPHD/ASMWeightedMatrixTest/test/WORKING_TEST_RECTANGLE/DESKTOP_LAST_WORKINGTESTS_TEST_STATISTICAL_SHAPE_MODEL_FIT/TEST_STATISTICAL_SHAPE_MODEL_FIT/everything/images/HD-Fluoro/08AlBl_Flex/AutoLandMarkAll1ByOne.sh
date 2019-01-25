'''for training set creation'''
#for i in {1..50};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/1-50/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/1-50/grey_image_"$i".jpg -n "$i" -m 2 ;done;

'''for initShape creation'''
#for i in {1..25};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../initShapes/ANDERST-ADAM/C5/51-75/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/51-100-havent_marked_yet/51-75/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
#for i in {1..21};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../initShapes/ANDERST-ADAM/C5/76-96/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/51-100-havent_marked_yet/76-96/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
#for i in {1..46};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/51-96/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/51-96/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;

'''for initShape creation'''
#for i in {1..25};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../initShapes/ANDERST-ADAM/C5/flex2_Cam_Inline_Cine1_cine_DistCorr-1-96/1-25/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/flex2_Cam_Inline_Cine1_cine_DistCorr-1-96/1-25/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
#for i in {1..25};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../initShapes/ANDERST-ADAM/C5/flex2_Cam_Inline_Cine1_cine_DistCorr-1-96/26-50/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/flex2_Cam_Inline_Cine1_cine_DistCorr-1-96/26-50/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;

#for i in {1..25};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/groundTruthflex2_1-50/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/flex2_Cam_Inline_Cine1_cine_DistCorr-1-96/1-25/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
#for i in {1..96};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/1-96/512x512/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/1-96/512x512/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;

#for i in {1..96};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/1-96/enhanced/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/1-96/enhanced/inverted/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
#cervical ADAM C5
for i in {1..41};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/ANDERST-ADAM/C5/1-96/enhanced/50-90/ -i ../../../../phdtest/images/ANDERST-ADAM/C5/1-96/enhanced/inverted/50-90/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;


#for i in {1..50};do python autoLandmarkContouring_shapeModify_TEST.py -p ../../../../phdtest/points/lastworking/1-50/trained/ -i ../../../../phdtest/images/lastworking/1-50/grey_image_$(($i+0)).jpg -n $(($i+0)) -m 2 ;done;
