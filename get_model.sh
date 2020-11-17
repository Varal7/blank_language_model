mkdir -p checkpoints
cd checkpoints
wget http://128.52.131.173:8000/yelp/neg.tgz -O neg.tgz
wget http://128.52.131.173:8000/yelp/pos.tgz -O pos.tgz
tar -xf neg.tgz
tar -xf pos.tgz
