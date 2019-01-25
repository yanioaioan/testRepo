
#!/bin/bash
for (( c=1; c<=50; c++ ))
do  
	echo "Python script called $c times"
	python cannyTest.py "$1""$c"".jpg"
done
