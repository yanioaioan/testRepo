#compare init shapes at the start of segmentation.. with final exportedPoints
#for i in {1..50}; do  python showLandmarksOnImage.py  ../../../../phdtest/points/all50/exportedPointsForEachStaticFrame/frame"$i"/ ../../../../phdtest/images/100-150/grey_image_"$i".jpg ../../../../initShapes/frame"$i"/ ; done;

#only ground truth export
#for i in {1..50}; do  python showLandmarksOnImage.py  ../../../../phdtest/points/all150/groundTruth-100-150/frame"$i"/ ../../../../phdtest/images/100-150/#grey_image_"$i".jpg ../../../../phdtest/points/all150/groundTruth-100-150/frame"$i"/ ; done;


#compare training set landmarks (considered ground truth).. with final exportedPoints
#for i in {1..50}; do  python showLandmarksOnImage.py  ../../../../phdtest/points/all50/exportedPointsForEachStaticFrame/frame"$i"/ ../../../../phdtest/images/100-150/grey_image_"$i".jpg ../../../../phdtest/points/all150/groundTruth-100-150/frame"$i"/ ; done;



#compare training set landmarks (considered ground truth).. with final exportedPoints
#for i in {1..50}; do  python showLandmarksOnImage.py  ../../../../phdtest/points/1-100/exportedPointsForEachStaticFrame/scale0_5_8/last/frame"$((100+i))"/ ../../../../phdtest/images/50-150/propernaming/grey_image_"$((50+i))".jpg ../../../../phdtest/points/all150/groundTruth-50-150/frame"$((50+i))"/ ; done;

#practically 50-90...renamed

#for i in {1..41}; do  python showLandmarksOnImageJustForPrinting.py  ../../../../phdtest/points/ANDERST-ADAM/C5/1-96/enhanced/50-90/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ../../../../phdtest/images/ANDERST-ADAM/C5/1-96/enhanced/inverted/50-90/targetFlex2Iinline/grey_image_"$((i))".jpg ../../../../phdtest/points/ANDERST-ADAM/C5/1-96/enhanced/50-90/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ; done;

#for i in {1..25}; do  python showLandmarksOnImageJustForPrinting.py  ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ../../../../phdtest/images/lastworking/1-50/target1-100_first_1_25/grey_image_"$((i))".jpg ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ; done;
#for i in {26..50}; do  python showLandmarksOnImageJustForPrinting.py  ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ../../../../phdtest/images/lastworking/1-50/target1-100_first_26_50/grey_image_"$((i))".jpg ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ; done;

#for i in {1..25}; do  python showLandmarksOnImageWithLandmarksOnContour.py  ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ../../../../phdtest/images/lastworking/1-50/target1-100_first_1_25/grey_image_"$((i))".jpg ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ; done;
for i in {26..50}; do  python showLandmarksOnImageWithLandmarksOnContour.py  ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ../../../../phdtest/images/lastworking/1-50/target1-100_first_26_50/grey_image_"$((i))".jpg ../../../../phdtest/points/lastworking/1-50/exportedPointsForEachStaticFrame/frame"$((i))"/scale0/ ; done;







