#usr/bin/bash
for i in {1..50}
do
		#echo  open frame$i/*pts
		counter=0

		
		while read line;do
		
		IFS=' ' read -r -a array <<< "$line"
		
		

		
		
		

	 
			if [ $counter -gt 2 ] && [ $counter -lt 46 ] ; then							
				
				#echo $counter"-->"${array[0]} ${array[1]}
		
				

				#var =`expr "${array[0]}" + 1`
				#echo ${array[0]}
				intpartX=${array[0]/.*} #convert float to int first source https://www.linuxquestions.org/questions/programming-9/convert-float-to-integer-in-bash-468503/
				echo $intpartX
				offsetByPixels=-8
				offsetedX=$((intpartX+offsetByPixels))
			
				echo ${offsetedX} ${array[1]} >> frame"$i"/"_$i".pts

			else
				echo $line >> frame"$i"/"_$i".pts #append the line as is
		
			fi	
		

	
		(( counter++ ))

		done < frame$i/"$i".pts
done

