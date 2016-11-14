#bash
for file in $1/*; 
    do cp "$file" $2/; 
done
