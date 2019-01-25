for i in {11..20};do python autoLandmarkContouring_shapeModify.py -p pts/ -i images/grey_image_"$i".jpg -n "$i" -m 2 ;done;
