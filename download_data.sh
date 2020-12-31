mkdir data
cd data

dir="http://people.csail.mit.edu/tianxiao/data"

wget $dir/yelp_blm.zip
unzip yelp_blm.zip
rm yelp_blm.zip

wget $dir/yahoo_blm.zip
unzip yahoo_blm.zip
rm yahoo_blm.zip
